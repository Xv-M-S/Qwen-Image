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

    def __init__(self):
        self.regional_mask = None
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "QwenDoubleStreamAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
        
    def RegionalQwenAttnProcessor2_0_call(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,  # Image stream
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
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
        if base_ratio is not None:
            hidden_states_base, encoder_hidden_states_base = self.RegionalQwenAttnProcessor2_0_call(
                attn = attn,
                hidden_states = hidden_states,
                encoder_hidden_states = encoder_hidden_states_base,
                encoder_hidden_states_mask = encoder_hidden_states_base_mask,
                attention_mask = attention_mask,
                image_rotary_emb=image_rotary_emb_base
            )

        # move regional mask to device
        if base_ratio is not None and 'regional_attention_mask' in joint_attention_kwargs:
            if self.regional_mask is not None:
                regional_mask = self.regional_mask.to(hidden_states.device)
            else:
                self.regional_mask = joint_attention_kwargs['regional_attention_mask']
                regional_mask = self.regional_mask.to(hidden_states.device)
        else:
            regional_mask = None

        hidden_states, encoder_hidden_states = self.RegionalQwenAttnProcessor2_0_call(
            attn = attn,
            hidden_states = hidden_states,
            encoder_hidden_states = encoder_hidden_states,
            encoder_hidden_states_mask = encoder_hidden_states_mask,
            attention_mask = regional_mask,
            image_rotary_emb=image_rotary_emb
        )


        if base_ratio is not None:
            if 'whole_regional_mask' in joint_attention_kwargs and joint_attention_kwargs["enable_whole_regional_mask"]:
                joint_attention_kwargs["whole_regional_mask"] = joint_attention_kwargs["whole_regional_mask"].to(hidden_states.device).to(hidden_states.dtype)
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
    """
        @torch.inference_mode() 是 PyTorch 提供的一个 装饰器（decorator），
        用于将函数或方法标记为“推理模式”（inference mode），即仅用于模型前向传播（forward pass），
        不进行梯度计算，也不构建计算图。
        它是 torch.no_grad() 的更严格、更高效的版本，专为推理（inference）场景设计。
    """
    @torch.inference_mode()
    def __call__(
        self,
        base_prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
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
        base_mask = torch.ones((height, width), device=device, dtype=self.transformer.dtype) # base mask uses the whole image mask
        base_inputs = [(base_mask, prompt_embeds)]

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

        # 6. Denoising loop
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                if i < mask_inject_steps:
                    chosen_prompt_embeds = regional_embeds
                    chosen_prompt_embeds_mask = regional_embeds_mask
                    base_ratio = attention_kwargs['base_ratio']
                else:
                    chosen_prompt_embeds = prompt_embeds
                    chosen_prompt_embeds_mask = prompt_embeds_mask
                    regional_txt_seq_lens = txt_seq_lens
                    base_ratio = None

                self._current_timestep = t
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
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
                        attention_kwargs={ # regional prompts kwargs -> add
                        # 'single_inject_blocks_interval': attention_kwargs['single_inject_blocks_interval'] if 'single_inject_blocks_interval' in attention_kwargs else len(self.transformer.transformer_blocks), 
                        'double_inject_blocks_interval': attention_kwargs['double_inject_blocks_interval'] if 'double_inject_blocks_interval' in attention_kwargs else len(self.transformer.transformer_blocks),
                        'regional_attention_mask': regional_attention_mask if base_ratio is not None else None,
                        "whole_regional_mask": attention_kwargs["whole_regional_mask"],  # 是否在相乘的时候只在mask上进行
                        "enable_whole_regional_mask": attention_kwargs["enable_whole_regional_mask"]
                        },
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