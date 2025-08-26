from diffusers import QwenImagePipeline
import torch
from typing import Union, List, Optional, Dict, Any, Callable, Tuple
from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipelineOutput
import numpy as np
from diffusers.pipelines.qwenimage.pipeline_qwenimage import calculate_shift, retrieve_timesteps, XLA_AVAILABLE
# import torch_xla.core.xla_model as xm
from diffusers.models.attention_processor import Attention
from diffusers.models.attention_dispatch import dispatch_attention_fn
import torch.nn.functional as F
from pipeline.attentionControl import AttentionStore
import re
from pipeline.gaussion_smoothing import GaussianSmoothing
from config.boxLossConfig import boxConfig
from util.tool import cost_time
from torchviz import make_dot
from util.visual import visualize_feature_activation, visualize_feature_channel, visualize_latent_map


def apply_rotary_emb_qwen(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, S, H, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio, OmniGen, CogView4 and Cosmos
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(1)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


class RegionalQwenImageAttnProcessor:
    """
    Attention processor for Qwen double-stream architecture, matching DoubleStreamLayerMegatron logic. This processor
    implements joint attention computation where text and image streams are processed together.
    """

    _attention_backend = None

    def __init__(self, attnstore= None):
        self.regional_mask = None
        self.attnstore = attnstore
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "QwenDoubleStreamAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
        


    def scaled_dot_product_attention_with_probs(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
        """
        手动实现 scaled dot-product attention，并返回 attention probs
        query, key, value: [B, H, S, D]
        参考官方伪代码: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        """
        L, S = query.size(-2), key.size(-2)
        scale = 1.0 / query.size(-1)**0.5

        # QK^T
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale  # [B, H, L, S]

        # 应用 causal mask
        # is_causal 参数在 scaled_dot_product_attention 函数中的作用是：
        # 启用因果注意力（Causal Attention）或称自回归掩码（Autoregressive Masking），
        # 它确保在解码过程中，每个位置只能关注到它之前（包括自身）的位置，而不能“看到”未来的信息。
        if is_causal:
            causal_mask = torch.triu(torch.ones(L, S, dtype=torch.bool, device=attn_scores.device), diagonal=1)
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

        # 应用用户提供的 mask
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask  # 注意：attn_mask 应为 float，-inf 表示遮挡

        # Softmax -> attention probs
        attn_probs = F.softmax(attn_scores, dim=-1)  # [B, H, L, S]

        # Dropout
        attn_probs = F.dropout(attn_probs, p=dropout_p, training=True)

        # @V
        attn_output = torch.matmul(attn_probs, value)  # [B, H, L, D]

        return attn_output, attn_probs  # ✅ 返回 probs
        
    def RegionalQwenAttnProcessor2_0_call(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,  # Image stream
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        index_block: str = None
    ) -> torch.FloatTensor:

        if encoder_hidden_states is None:
            raise ValueError("QwenDoubleStreamAttnProcessor2_0 requires encoder_hidden_states (text stream)")

        seq_txt = encoder_hidden_states.shape[1]

        # Compute QKV for image stream
        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)

        # Compute QKV for text stream
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        # Reshape for multi-head attention
        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))

        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        # Apply QK normalization
        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        # Apply RoPE
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

        # Concatenate for joint attention: [text, image]
        joint_query = torch.cat([txt_query, img_query], dim=1).permute(0,2,1,3)  # dim=2: seq_len
        joint_key = torch.cat([txt_key, img_key], dim=1).permute(0,2,1,3)
        joint_value = torch.cat([txt_value, img_value], dim=1).permute(0,2,1,3)



        joint_hidden_states, attn_probs = self.scaled_dot_product_attention_with_probs(
            query=joint_query,
            key=joint_key,
            value=joint_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False
        )

        # Store attention for box loss
        if self.attnstore is not None and boxConfig.switch_box_loss and index_block in boxConfig.train_layer:
            # Extract image-to-text cross attention: img_query vs txt_key
            # joint_query: [txt_query, img_query] -> img_query starts at seq_txt
            # joint_key:   [txt_key,   img_key]   -> txt_key ends at seq_txt
            img_txt_attn = attn_probs[:, :, seq_txt:, :seq_txt]  # [B, H, S_img, S_txt]
            img_txt_attn = img_txt_attn.mean(dim=1)  # Average over heads -> [B, S_img, S_txt]
            is_cross = True
            self.attnstore(img_txt_attn, is_cross)
            if img_txt_attn.shape[2] == boxConfig.text_len and boxConfig.visual_attention_map:
                # no viusal of negative prompt
                visualize_feature_activation(img_txt_attn.clone().detach(), index_block)
                visualize_feature_channel(img_txt_attn.clone().detach(), index_block)

        # Reshape back
        joint_hidden_states = joint_hidden_states.transpose(1, 2).flatten(2, 3)  # [B, S_joint, H*D]
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        # Split
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]

        # Output projections
        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)

        txt_attn_output = attn.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output
    def RegionalQwenAttnProcessor2_0_call_old(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,  # Image stream
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        index_block: str = None
    ) -> torch.FloatTensor:
        if encoder_hidden_states is None:
            raise ValueError("QwenDoubleStreamAttnProcessor2_0 requires encoder_hidden_states (text stream)")


        seq_txt = encoder_hidden_states.shape[1]

        # Compute QKV for image stream (sample projections)
        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)

        # Compute QKV for text stream (context projections)
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        # Reshape for multi-head attention
        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))

        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        # Apply QK normalization
        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        # Apply RoPE
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

        # Concatenate for joint attention
        # Order: [text, image]
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        # Compute joint attention
        joint_hidden_states = dispatch_attention_fn(
            joint_query,
            joint_key,
            joint_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
        )

        # 使用不加速的方式重新计算一遍 attention socre
        if self.attnstore is not None and boxConfig.switch_box_loss and index_block in boxConfig.train_layer:
            score_query = hidden_states # img 
            score_key = encoder_hidden_states # txt
            attention_probs = attn.get_attention_scores(score_query, score_key, attention_mask=None)
            is_cross = encoder_hidden_states is not None
            self.attnstore(attention_probs, is_cross)


        # Reshape back
        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        # Split attention outputs back
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]  # Text part
        img_attn_output = joint_hidden_states[:, seq_txt:, :]  # Image part

        # Apply output projections
        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)  # dropout

        txt_attn_output = attn.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output
          

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,  # Image stream
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream
        encoder_hidden_states_mask: torch.FloatTensor = None, 
        encoder_hidden_states_base: torch.FloatTensor = None, # added
        encoder_hidden_states_base_mask: torch.FloatTensor = None, # added
        base_ratio: float = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        image_rotary_emb_base: Optional[torch.Tensor] = None, # added
        joint_attention_kwargs: Optional[Dict[str, Any]] = None, # added
    ) -> torch.FloatTensor:
        index_block = joint_attention_kwargs["index_block"] if joint_attention_kwargs is not None else None
        if base_ratio is not None:
            hidden_states_base, encoder_hidden_states_base = self.RegionalQwenAttnProcessor2_0_call(
                attn = attn,
                hidden_states = hidden_states,
                encoder_hidden_states = encoder_hidden_states_base,
                encoder_hidden_states_mask = encoder_hidden_states_base_mask,
                attention_mask = attention_mask,
                image_rotary_emb=image_rotary_emb_base,
                index_block = index_block
            )

        # move regional mask to device
        if base_ratio is not None and 'regional_attention_mask' in joint_attention_kwargs:
            if self.regional_mask is not None:
                if  self.regional_mask.device != hidden_states.device:
                    regional_mask = self.regional_mask.to(hidden_states.device)
                else:
                    regional_mask = self.regional_mask
            else:
                # regional_mask = self.regional_mask.to(hidden_states.device)
                # avoid duplicate move to device
                self.regional_mask = joint_attention_kwargs['regional_attention_mask']
                regional_mask = self.regional_mask
        else:
            regional_mask = None

        hidden_states, encoder_hidden_states = self.RegionalQwenAttnProcessor2_0_call(
            attn = attn,
            hidden_states = hidden_states,
            encoder_hidden_states = encoder_hidden_states,
            encoder_hidden_states_mask = encoder_hidden_states_mask,
            attention_mask = regional_mask,
            image_rotary_emb=image_rotary_emb,
            index_block = index_block
        )


        if base_ratio is not None:
            if 'whole_regional_mask' in joint_attention_kwargs and joint_attention_kwargs["enable_whole_regional_mask"]:
                joint_attention_kwargs["whole_regional_mask"] = joint_attention_kwargs["whole_regional_mask"]  # .to(hidden_states.device).to(hidden_states.dtype)
                # mask merge method 1: non_regional_area 100 per use base_prompt
                hidden_states = hidden_states * joint_attention_kwargs["whole_regional_mask"] *(1 - base_ratio) + hidden_states_base * joint_attention_kwargs["whole_regional_mask"] * base_ratio
                hidden_states = hidden_states + hidden_states_base * (1 - joint_attention_kwargs["whole_regional_mask"])

                # apply regional control to hidden_states_base
                # hidden_states will be used for generate hidden_states_base and hidden_states for next step

                # mask merge method 2: regional_area use 100 per regional_control and add base_ratio non_regional_control
                # hidden_states = hidden_states * joint_attention_kwargs["whole_regional_mask"] * (1 - base_ratio) + hidden_states_base * base_ratio
                # hidden_states = hidden_states / (joint_attention_kwargs["whole_regional_mask"] +  base_ratio)  # avoid division by zero
            else:
                # merge hidden_states and hidden_states_base
                hidden_states = hidden_states*(1-base_ratio) + hidden_states_base*base_ratio
            return hidden_states, encoder_hidden_states, encoder_hidden_states_base
        else: # both regional and base input are base prompts, skip the merge
            return hidden_states, encoder_hidden_states, encoder_hidden_states



        

class RegionalQwenImagePipeline(QwenImagePipeline):

    def get_token_index(self, prompt, quote_prompt):
        # 对基础提示进行 tokenization
        # 步骤 1: 提取所有双引号内的内容（保留原样）
        print("🔍 提取的引号内容:")
        quoted_texts = re.findall(r'"(.*?)"', prompt)
        for i, text in enumerate(quoted_texts):
            print(f"  [{i}] {repr(text)}")
        
        # 步骤 2: Tokenize 整个 prompt
        tokens = self.tokenizer.tokenize(prompt)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        print(f"\n📊 总 token 数: {len(tokens)}")
        print("🔤 前 20 个 tokens:", tokens[:100])

        # 步骤 3: 对每个引号内容，查找其在 token stream 中的位置
        print("\n📌 引号内容在 tokenizer 中的位置:")
        quote_to_token_positions = {}

        for idx, quote in enumerate(quoted_texts):
            print(f"\n--- 处理引号内容 [{idx}]: {repr(quote)} ---")

            # 将引号内容单独 tokenize
            quote_tokens = self.tokenizer.tokenize(quote)
            quote_token_ids = self.tokenizer.convert_tokens_to_ids(quote_tokens)

            print(f"  Tokenized 子串: {quote_tokens}")
            print(f"  Token IDs: {quote_token_ids}")

            # 在完整 token stream 中查找匹配
            found = False
            positions = None
            for start_idx in range(len(token_ids) - len(quote_token_ids) + 1):
                end_idx = start_idx + len(quote_token_ids)
                if token_ids[start_idx:end_idx] == quote_token_ids:
                    print(f"  ✅ 匹配位置: 起始索引 = {start_idx}, 结束索引 = {end_idx - 1} (token 范围: [{start_idx}:{end_idx}])")
                    # 可选：验证一下还原的文本
                    reconstructed = self.tokenizer.decode(token_ids[start_idx:end_idx])
                    print(f"  🔁 重建文本: {repr(reconstructed)}")
                    found = True
                    positions = list(range(start_idx, end_idx))
                    break

            if positions is not None:
                quote_to_token_positions[quote] = positions
                print(f"  ✅ 匹配成功，token 位置: {positions}")
                print(f"  🔄 从 tokens 重建: {repr(self.tokenizer.decode(token_ids[positions[0]:positions[-1]+1]))}")
            else:
                print(f"  ❌ 未找到匹配")
                quote_to_token_positions[quote] = []
        
        # ======================
        # 6. 最终结果：每个引号内容对应的 token 位置集合
        # ======================
        print("\n" + "="*60)
        print("✅ 每个引号内容在 token 序列中的位置集合：")
        print("="*60)
        for quote, positions in quote_to_token_positions.items():
            print(f"""
        引号内容: "{quote}"
        位置集合: {positions}
        长度: {len(positions)} tokens
        """)
        
        # 从 Python 3.7 开始，dict（字典）保证保持插入顺序。
        return quote_to_token_positions
    
    def aggregate_attention(
                        self,
                        attention_store: AttentionStore,
                        is_cross: bool,
                        shape: Tuple[int, int],
                        select: int # select 为什么要设置为0，需要探究？
                    ) -> torch.Tensor:
        """ Aggregates the attention across the different layers and heads at the specified resolution. """
        out = []
        attention_maps = attention_store.get_store_attention()
        # print(attention_maps)

        for item in attention_maps[f"{'cross' if is_cross else 'self'}"]:
            cross_maps = item.reshape(1, -1, int(shape[0]/shape[2]), int(shape[1]/shape[2]), item.shape[-1])[select]
            out.append(cross_maps)
        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out
    

    def _compute_max_attention_per_index(self,
                                        attention_maps: torch.Tensor,
                                        indices_to_alter: List[int],
                                        smooth_attentions: bool = False,
                                        shape: Optional[Tuple[int, int, int]] = None,
                                        sigma: float = 0.5,
                                        kernel_size: int = 3,
                                        bbox: List[int] = None,
                                        ) -> List[torch.Tensor]:
        """ Computes the maximum attention value for each of the tokens we wish to alter. """
        # QwenVL 截断了 eot token,第一个便是有效字符 -> 旧实现见 _compute_max_attention_per_index_old
        attention_for_text = attention_maps
        # attention_for_text = attention_for_text * 100
        # attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)
        # attention_for_text = attention_for_text * 100
        # attention_sum = attention_for_text.sum(dim=-1)

        # Extract the maximum values
        max_indices_list_fg = []
        # for fit the old implementation, define but not used
        max_indices_list_bg = []
        dist_x = []
        dist_y = []

        height, width, scale_factor = shape

        cnt = 0
        """
            情况不一样 -> boxdiff是一个框只有一个token描述,而我们一个框有多个token描述,此处有待改进
        """
        for cnt, indices in enumerate(indices_to_alter):
            image_list = []

            for i in indices:
                image = attention_for_text[:, :, i]
                image_list.append(image)

            image_mean = torch.stack(image_list).mean(dim=0)
            # 基于极大极小值进行归一化
            image_mean = (image_mean - image_mean.min()) / (image_mean.max() - image_mean.min() + + 1e-8)

            box = [max(round(b/scale_factor), 0) for b in bbox[cnt]]
            x1, y1, x2, y2 = box

            # coordinates to masks
            obj_mask = torch.zeros_like(image_mean)
            ones_mask = torch.ones([y2 - y1, x2 - x1], dtype=obj_mask.dtype).to(obj_mask.device)
            obj_mask[y1:y2, x1:x2] = ones_mask
            bg_mask = 1 - obj_mask

            if smooth_attentions:
                smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).cuda()
                input = F.pad(image_mean.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
                image = smoothing(input).squeeze(0).squeeze(0)

            # Inner-Box constraint
            inner_pix_num = obj_mask.sum()
            inner_pix_avg = (image_mean * obj_mask).sum() /inner_pix_num


            # Outer-Box constraint
            outer_pix_num = bg_mask.sum()
            outer_pix_avg = (image_mean * bg_mask).sum() /outer_pix_num

            diff = inner_pix_avg - outer_pix_avg
            loss = - torch.log(torch.sigmoid(diff) + 1e-8)

            max_indices_list_fg.append(loss)

        return max_indices_list_fg, max_indices_list_bg, dist_x, dist_y

            
    def _compute_max_attention_per_index_old(self,
                                        attention_maps: torch.Tensor,
                                        indices_to_alter: List[int],
                                        smooth_attentions: bool = False,
                                        shape: Optional[Tuple[int, int, int]] = None,
                                        sigma: float = 0.5,
                                        kernel_size: int = 3,
                                        bbox: List[int] = None,
                                        ) -> List[torch.Tensor]:
        """ Computes the maximum attention value for each of the tokens we wish to alter. """
        # last_idx = -1
        # if normalize_eot:
        #     prompt = self.prompt
        #     if isinstance(self.prompt, list):
        #         prompt = self.prompt[0]
        #     last_idx = len(self.tokenizer(prompt)['input_ids']) - 1
        # attention_for_text = attention_maps[:, :, 1:last_idx]
        # QwenVL 截断了 eot token,第一个便是有效字符
        attention_for_text = attention_maps
        attention_for_text = attention_for_text * 100
        attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        # Shift indices since we removed the first token
        # indices_to_alter = [index - 1 for index in indices_to_alter]
        indices_to_alter = [[index - 1 for index in sublist] for sublist in indices_to_alter]

        # Extract the maximum values
        max_indices_list_fg = []
        max_indices_list_bg = []
        dist_x = []
        dist_y = []

        height, width, scale_factor = shape

        cnt = 0
        """
            情况不一样 -> boxdiff是一个框只有一个token描述,而我们一个框有多个token描述,此处有待改进
        """
        for cnt, indices in enumerate(indices_to_alter):

            for i in indices:
                image = attention_for_text[:, :, i]

                box = [max(round(b/scale_factor), 0) for b in bbox[cnt]]
                x1, y1, x2, y2 = box

                # coordinates to masks
                obj_mask = torch.zeros_like(image)
                ones_mask = torch.ones([y2 - y1, x2 - x1], dtype=obj_mask.dtype).to(obj_mask.device)
                obj_mask[y1:y2, x1:x2] = ones_mask
                bg_mask = 1 - obj_mask

                if smooth_attentions:
                    smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).cuda()
                    input = F.pad(image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
                    image = smoothing(input).squeeze(0).squeeze(0)

                # Inner-Box constraint
                k = (obj_mask.sum() * boxConfig.P).long()
                max_indices_list_fg.append((image * obj_mask).reshape(-1).topk(k)[0].mean())

                # Outer-Box constraint
                k = (bg_mask.sum() * boxConfig.P).long()
                max_indices_list_bg.append((image * bg_mask).reshape(-1).topk(k)[0].mean())

                # Corner Constraint
                gt_proj_x = torch.max(obj_mask, dim=0)[0]
                gt_proj_y = torch.max(obj_mask, dim=1)[0]
                corner_mask_x = torch.zeros_like(gt_proj_x)
                corner_mask_y = torch.zeros_like(gt_proj_y)

                # create gt according to the number config.L
                N = gt_proj_x.shape[0]
                corner_mask_x[max(box[0] - boxConfig.L, 0): min(box[0] + boxConfig.L + 1, N)] = 1.
                corner_mask_x[max(box[2] - boxConfig.L, 0): min(box[2] + boxConfig.L + 1, N)] = 1.
                corner_mask_y[max(box[1] - boxConfig.L, 0): min(box[1] + boxConfig.L + 1, N)] = 1.
                corner_mask_y[max(box[3] - boxConfig.L, 0): min(box[3] + boxConfig.L + 1, N)] = 1.
                dist_x.append((F.l1_loss(image.max(dim=0)[0], gt_proj_x, reduction='none') * corner_mask_x).mean())
                dist_y.append((F.l1_loss(image.max(dim=1)[0], gt_proj_y, reduction='none') * corner_mask_y).mean())

        return max_indices_list_fg, max_indices_list_bg, dist_x, dist_y

    @cost_time
    def _aggregate_and_get_max_attention_per_token(self, 
                                                    attention_store: AttentionStore,
                                                    indices_to_alter:Dict[str, List[int]],
                                                    gaussian_smoothing_kwargs:Dict[str, Any],
                                                    shape: Tuple[int, int, int],
                                                    bbox:List[List[int]]
                                                        ):
        """ Aggregates the attention for each token and computes the max activation value for each token to alter. """
        attention_maps = self.aggregate_attention(
                attention_store=attention_store,
                shape=shape, 
                is_cross=True,
                select=0 # why 0
            )
        values_list = [indices_to_alter[key] for key in indices_to_alter]
        max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y = self._compute_max_attention_per_index(
            attention_maps=attention_maps,
            indices_to_alter=values_list,
            smooth_attentions=gaussian_smoothing_kwargs.get("smooth_attentions", False),
            sigma=gaussian_smoothing_kwargs.get("sigma", 0.5),
            kernel_size=gaussian_smoothing_kwargs.get("kernel_size", 5),
            shape=shape,
            bbox=bbox
        )
        return max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y
    

    @staticmethod
    @cost_time
    def _compute_loss_old(max_attention_per_index_fg: List[torch.Tensor], max_attention_per_index_bg: List[torch.Tensor],
                      dist_x: List[torch.Tensor], dist_y: List[torch.Tensor], return_losses: bool = False) -> torch.Tensor:
        """ Computes the attend-and-excite loss using the maximum attention value for each token. """
        losses_fg = [max(0, 1. - curr_max) for curr_max in max_attention_per_index_fg]
        losses_bg = [max(0, curr_max) for curr_max in max_attention_per_index_bg]
        loss = sum(losses_fg) + sum(losses_bg) + sum(dist_x) + sum(dist_y)
        print(f"losses_fg: {max(losses_fg)} loss_bg:{max(losses_bg)} , loss:{loss}")
        if return_losses:
            return max(losses_fg), losses_fg
        else:
            return max(losses_fg), loss
        
    def _compute_loss(self, max_attention_per_index_fg: List[torch.Tensor], max_attention_per_index_bg: List[torch.Tensor],
                      dist_x: List[torch.Tensor], dist_y: List[torch.Tensor], return_losses: bool = False) -> torch.Tensor:
        """ Computes the attend-and-excite loss using the maximum attention value for each token. """
        loss = sum(max_attention_per_index_fg)
        print(f"loss: {loss}")
        return loss, loss
    
    @staticmethod
    @cost_time
    def _update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float) -> torch.Tensor:
        """ Update the latent according to the computed loss. """
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
        # 1. 全局归一化（最推荐）
        # grad_cond = grad_cond / (grad_cond.norm() + 1e-8)
        # 2. 最大最小值归一化
        # grad_cond = (grad_cond - grad_cond.min()) / (grad_cond.max() - grad_cond.min() + 1e-8)
        # 3. 均值归一化 -> 这种归一化能放大梯度，同时不改变梯度的分布（核心思想：不改变梯度的比例系数、分布的情况下，合理的放大梯度）
        grad_cond = (grad_cond - grad_cond.mean()) / (grad_cond.std() + 1e-8)
        latents = latents - step_size * grad_cond
        return latents
        

    def _perform_iterative_refinement_step(self,
                                           latents: torch.Tensor,
                                           encoder_hidden_states_mask: torch.Tensor,
                                           encoder_hidden_states: torch.Tensor,
                                           img_shapes: Tuple,
                                           regional_txt_seq_lens: int,
                                           indices_to_alter: List[int],
                                           loss_fg: torch.Tensor,
                                           threshold: float,
                                           attention_store: AttentionStore,
                                           step_size: float,
                                           timestep: torch.Tensor,
                                           guidance: torch.Tensor,
                                           max_refinement_steps: int = 20
                                           ):
        """
        Performs the iterative latent refinement introduced in the paper. Here, we continuously update the latent
        code according to our loss objective until the given threshold is reached for all tokens.
        """
        iteration = 0
        target_loss = max(0, 1. - threshold)
        while loss_fg > target_loss:
            iteration += 1

            latents = latents.clone().detach().requires_grad_(True)
            self.set_train_transform_layer(boxConfig.train_layer)
            noise_pred_text = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=encoder_hidden_states_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        img_shapes=img_shapes,
                        # txt_seq_lens=negative_txt_seq_lens,
                        regional_txt_seq_lens=regional_txt_seq_lens,
                        attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                    )[0]
            self.transformer.zero_grad()

            # Get max activation value for each subject token
            max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y = self._aggregate_and_get_max_attention_per_token(
                attention_store=attention_store,
                indices_to_alter=indices_to_alter,
                gaussian_smoothing_kwargs=self._gaussian_smoothing_kwargs,
                shape = (self._height ,self._width ,self.vae_scale_factor*2),
                bbox=self.attention_kwargs.get("regional_boxes")
            )

            loss_fg, losses_fg = self._compute_loss(max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y, return_losses=True)

            if loss_fg != 0:  # 此处算出来的梯度特别小，导致loss_fg基本没更新
                latents = self._update_latent(latents, loss_fg, step_size)

        
            if iteration >= max_refinement_steps:
                print(f'\t Exceeded max number of iterations ({max_refinement_steps})! ')
                break
        return latents
    
    @cost_time
    def set_train_transform_layer(self, train_transform_layer):
        for name, param in self.transformer.named_parameters():
            param.requires_grad = False
            if 'transformer_blocks' in name:
                layer_index = int(name.split(".")[1])
                if layer_index in train_transform_layer:
                    param.requires_grad = True

    """
        @torch.inference_mode() 是 PyTorch 提供的一个 装饰器（decorator），
        用于将函数或方法标记为“推理模式”（inference mode），即仅用于模型前向传播（forward pass），
        不进行梯度计算，也不构建计算图。
        它是 torch.no_grad() 的更严格、更高效的版本，专为推理（inference）场景设计。
    """
    # @torch.inference_mode() 会抑制梯度计算，所以此处用@torch.no_grad()
    # @torch.inference_mode()
    @torch.no_grad()
    def __call__(
        self,
        base_prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        attention_store: AttentionStore = None,
        true_cfg_scale: float = 4.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        mask_inject_steps: int = 5,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 1.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        gaussian_smoothing_kwargs: Optional[Dict[str, Any]] = None, # added ,用于attention map 的高斯平滑
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
                not greater than `1`).
            true_cfg_scale (`float`, *optional*, defaults to 1.0):
                When > 1.0 and a provided `negative_prompt`, enables true classifier-free guidance.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 3.5):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.qwenimage.QwenImagePipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.qwenimage.QwenImagePipelineOutput`] or `tuple`:
            [`~pipelines.qwenimage.QwenImagePipelineOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is a list with the generated images.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            base_prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False
        self._gaussian_smoothing_kwargs = gaussian_smoothing_kwargs if gaussian_smoothing_kwargs is not None else {}
        self._height = height
        self._width = width

        # get quote prompt token index
        if isinstance(base_prompt, str) and '"' in base_prompt:
            quote_to_token_positions = self.get_token_index(base_prompt, quote_prompt=True)
            print("🔗 引号内容对应的 token 位置:", quote_to_token_positions)
        else:
            quote_to_token_positions = None

        # 2. Define call parameters
        if base_prompt is not None and isinstance(base_prompt, str):
            batch_size = 1
        elif base_prompt is not None and isinstance(base_prompt, list):
            batch_size = len(base_prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        prompt_embeds, prompt_embeds_mask = self.encode_prompt( # 在只输入文本的情况下，可以看成是使用Qwen-7B-Chat进行文本编码
            prompt=base_prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        boxConfig.text_len = prompt_embeds.shape[1]
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        # added: define base mask and inputs
        # base_mask = torch.ones((height, width), device=device, dtype=self.transformer.dtype) # base mask uses the whole image mask
        # base_inputs = [(base_mask, prompt_embeds)]

        # added: encode regional prompts,define regional inputs
        regional_inputs = []
        if 'regional_prompts' in attention_kwargs and 'regional_masks' in attention_kwargs:
            for regional_prompt, regional_mask in zip(attention_kwargs['regional_prompts'], attention_kwargs['regional_masks']):
                regional_prompt_embeds, regional_prompt_embeds_masks = self.encode_prompt(
                    prompt=regional_prompt,
                    prompt_embeds=None,
                    prompt_embeds_mask=None,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length
                )
                regional_inputs.append((regional_mask, regional_prompt_embeds, regional_prompt_embeds_masks))

        ## added: prepare masks for regional control
        conds = []
        cond_masks = []
        masks = []
        each_prompt_seq_len = [] 
        H, W = height//(self.vae_scale_factor)//2, width//(self.vae_scale_factor)//2
        hidden_seq_len = H * W

        # prepare base ration regional masks
        if attention_kwargs is not None and attention_kwargs["enable_whole_regional_mask"]:
            attention_kwargs["whole_regional_mask"] = torch.nn.functional.interpolate(attention_kwargs["whole_regional_mask"][None, None, :, :], (H, W), mode='nearest-exact').flatten().unsqueeze(1)
        
        for mask, cond, cond_mask in regional_inputs:
            if mask is not None: # resize regional masks to image size, the flatten is to match the seq len
                mask = torch.nn.functional.interpolate(mask[None, None, :, :], (H, W), mode='nearest-exact').flatten().unsqueeze(1).repeat(1, cond.size(1))
            else:
                mask = torch.ones((H*W, cond.size(1))).to(device=cond.device)
            masks.append(mask)
            conds.append(cond)
            cond_masks.append(cond_mask)
            each_prompt_seq_len.append(cond.shape[1])
        regional_embeds = torch.cat(conds, dim=1)
        regional_embeds_mask = torch.cat(cond_masks, dim=1)
        encoder_seq_len = regional_embeds.shape[1]

        # initialize attention mask
        regional_attention_mask = torch.zeros(
            (encoder_seq_len + hidden_seq_len, encoder_seq_len + hidden_seq_len),
            device=masks[0].device,
            dtype=torch.bool
        )
        num_of_regions = len(masks)

        # initialize self-attended mask
        self_attend_masks = torch.zeros((hidden_seq_len, hidden_seq_len), device=masks[0].device, dtype=torch.bool)

        # initialize union mask
        union_masks = torch.zeros((hidden_seq_len, hidden_seq_len), device=masks[0].device, dtype=torch.bool)

        # handle each mask
        seq_len_begin = 0
        seq_len_end = 0
        for i in range(num_of_regions):
            # caculate the begin and end of the current region
            seq_len_begin = seq_len_end
            seq_len_end = seq_len_begin + each_prompt_seq_len[i]

            # txt attends to itself
            # regional_attention_mask[i*each_prompt_seq_len:(i+1)*each_prompt_seq_len, i*each_prompt_seq_len:(i+1)*each_prompt_seq_len] = True
            regional_attention_mask[seq_len_begin:seq_len_end, seq_len_begin:seq_len_end] = True

            # txt attends to corresponding regional img
            # regional_attention_mask[i*each_prompt_seq_len:(i+1)*each_prompt_seq_len, encoder_seq_len:] = masks[i].transpose(-1, -2)
            regional_attention_mask[seq_len_begin:seq_len_end, encoder_seq_len:] = masks[i].transpose(-1, -2)

            # regional img attends to corresponding txt
            # regional_attention_mask[encoder_seq_len:, i*each_prompt_seq_len:(i+1)*each_prompt_seq_len] = masks[i]
            regional_attention_mask[encoder_seq_len:, seq_len_begin:seq_len_end] = masks[i]

            # regional img attends to corresponding regional img
            img_size_masks = masks[i][:, :1].repeat(1, hidden_seq_len)
            img_size_masks_transpose = img_size_masks.transpose(-1, -2)
            self_attend_masks = torch.logical_or(self_attend_masks, 
                                                    torch.logical_and(img_size_masks, img_size_masks_transpose))

            # update union
            union_masks = torch.logical_or(union_masks, 
                                            torch.logical_or(img_size_masks, img_size_masks_transpose))

        background_masks = torch.logical_not(union_masks)

        background_and_self_attend_masks = torch.logical_or(background_masks, self_attend_masks)

        regional_attention_mask[encoder_seq_len:, encoder_seq_len:] = background_and_self_attend_masks
        ## added : done prepare masks for regional control


        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        img_shapes = [(1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2)] * batch_size

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        scale_range = np.linspace(boxConfig.scale_range[0], boxConfig.scale_range[1], self._num_timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
        )
        regional_txt_seq_lens = regional_embeds_mask.sum(dim=1).tolist() if regional_embeds_mask is not None else None

        # handle infer attention mask
        regional_attention_mask = regional_attention_mask.to(device)
        infer_attention_kwargs = {
            'double_inject_blocks_interval': attention_kwargs['double_inject_blocks_interval'] if 'double_inject_blocks_interval' in attention_kwargs else len(self.transformer.transformer_blocks),
            "whole_regional_mask": attention_kwargs["whole_regional_mask"].to(device).to(latents.dtype),  # 操作是否局限在mask内
            "enable_whole_regional_mask": attention_kwargs["enable_whole_regional_mask"]
        }

        # add some args for visualization
        boxConfig.text_index = quote_to_token_positions
        boxConfig.bbox = attention_kwargs.get("regional_boxes")


        # 6. Denoising loop
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                boxConfig.now_step = i

                self._current_timestep = t
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                # 基于局部梯度更新latents，使得初始latents的布局更符合区域提示的要求
                if i < boxConfig.max_iter_to_alter:
                    boxConfig.switch_box_loss = True
                    with torch.enable_grad():
                        latents = latents.clone().detach().requires_grad_(True)

                        # train all layers has no such big memory cost
                        self.set_train_transform_layer(boxConfig.train_layer)

                        # Forward pass of denoising with text conditioning
                        noise_pred_text = self.transformer(
                            hidden_states=latents,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=prompt_embeds_mask,
                            encoder_hidden_states=prompt_embeds,
                            img_shapes=img_shapes,
                            # txt_seq_lens=negative_txt_seq_lens,
                            regional_txt_seq_lens=txt_seq_lens,
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                        )[0]

                        self.transformer.zero_grad()

                        # Get max activation value for each subject token
                        max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y = self._aggregate_and_get_max_attention_per_token(
                            attention_store=attention_store,
                            indices_to_alter=quote_to_token_positions,
                            gaussian_smoothing_kwargs=gaussian_smoothing_kwargs,
                            shape = (height,width,self.vae_scale_factor*2),
                            bbox=attention_kwargs.get("regional_boxes")
                        )

                        # Perform gradient update # 此处是几个局部的差值算了一个总的loss,然后该loss应用于全局，这样是否合理？存在问题，需要改进
                        if i < boxConfig.max_iter_to_alter:
                            loss_fg, loss = self._compute_loss(max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y)
                            if loss != 0:
                                latents = self._update_latent(latents=latents, loss=loss_fg, # 原实现此处用loss
                                                                step_size=boxConfig.scale_factor * np.sqrt(scale_range[i]))

                        # Refinement from attend-and-excite (not necessary)
                        if True:

                            # loss_fg, loss = self._compute_loss(max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y)

                            if i in boxConfig.thresholds.keys() and loss_fg > 1. - boxConfig.thresholds.get(i) and boxConfig.refine:
                                del noise_pred_text
                                torch.cuda.empty_cache()
                                latents = self._perform_iterative_refinement_step(
                                    latents=latents,
                                    encoder_hidden_states_mask=prompt_embeds_mask,
                                    encoder_hidden_states=prompt_embeds,
                                    img_shapes=img_shapes,
                                    regional_txt_seq_lens=txt_seq_lens,
                                    indices_to_alter=quote_to_token_positions,
                                    loss_fg=loss_fg,
                                    threshold=boxConfig.thresholds.get(i),
                                    attention_store=attention_store,
                                    step_size= boxConfig.scale_factor * np.sqrt(scale_range[i]),
                                    timestep=timestep,
                                    guidance=guidance,  # use a larger guidance scale for refinement
                                    max_refinement_steps=boxConfig.max_refinement_steps
                                )
                        
                        
                        boxConfig.switch_box_loss = False

                       



                if self.interrupt:
                    continue

                if i < mask_inject_steps:
                    chosen_prompt_embeds = regional_embeds
                    chosen_prompt_embeds_mask = regional_embeds_mask
                    base_ratio = attention_kwargs['base_ratio']
                    infer_attention_kwargs["regional_attention_mask"] = regional_attention_mask
                else:
                    chosen_prompt_embeds = prompt_embeds
                    chosen_prompt_embeds_mask = prompt_embeds_mask
                    regional_txt_seq_lens = txt_seq_lens
                    base_ratio = None

                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=chosen_prompt_embeds_mask,
                        encoder_hidden_states=chosen_prompt_embeds, # regional_embeds or base prompt_embeds -> change
                        encoder_hidden_states_base_mask=prompt_embeds_mask, # base prompt mask -> add
                        encoder_hidden_states_base=prompt_embeds, # base prompt embeds -> add
                        base_ratio=base_ratio, # base ratio for regional control -> add
                        img_shapes=img_shapes,
                        txt_seq_lens=txt_seq_lens,
                        regional_txt_seq_lens=regional_txt_seq_lens,
                        attention_kwargs=infer_attention_kwargs,
                        return_dict=False,
                    )[0]

                if do_true_cfg:
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latents,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=negative_prompt_embeds_mask,
                            encoder_hidden_states=negative_prompt_embeds,
                            img_shapes=img_shapes,
                            # txt_seq_lens=negative_txt_seq_lens,
                            regional_txt_seq_lens=negative_txt_seq_lens,
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                        )[0]
                    comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_pred = comb_pred * (cond_norm / noise_norm)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                    
                if boxConfig.visual_middle_res:
                    visualize_latent_map(self, latents.clone().detach(), height, width, i)

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return QwenImagePipelineOutput(images=image)