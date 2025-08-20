from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel, USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers, logger
from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from diffusers.models.attention_processor import AttentionProcessor




class RegionalQwenImageTransformer(QwenImageTransformer2DModel):

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

     # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        base_ratio: float = None, # added for regional control
        encoder_hidden_states_base: torch.Tensor = None, # added
        encoder_hidden_states_base_mask: torch.Tensor = None, # added
        timestep: torch.LongTensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        txt_seq_lens: Optional[List[int]] = None,
        regional_txt_seq_lens: Optional[List[int]] = None,  # added for regional control
        guidance: torch.Tensor = None,  # TODO: this should probably be removed
        attention_kwargs: Optional[Dict[str, Any]] = None, # used for regional control
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        The [`QwenTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            encoder_hidden_states_mask (`torch.Tensor` of shape `(batch_size, text_sequence_length)`):
                Mask of the input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        hidden_states = self.img_in(hidden_states)

        additional_kwargs = {}
        if base_ratio is not None:
            additional_kwargs["regional_attention_mask"] = attention_kwargs['regional_attention_mask']
        # 是否启用区域mask技术
        if attention_kwargs["enable_whole_regional_mask"]:
            additional_kwargs["whole_regional_mask"] = attention_kwargs["whole_regional_mask"]
            additional_kwargs["enable_whole_regional_mask"] = attention_kwargs["enable_whole_regional_mask"]
        additional_kwargs["hidden_seq_len"] = hidden_states.shape[1]

        # when using controlnet, only one prompt is provided, so we need to consider the case
        if encoder_hidden_states_base is not None:  
            encoder_hidden_states_base = self.txt_norm(encoder_hidden_states_base)
            encoder_hidden_states_base = self.txt_in(encoder_hidden_states_base)
            additional_kwargs["encoder_seq_len_base"] = encoder_hidden_states_base.shape[1]
        else:
            encoder_hidden_states_base = None

        timestep = timestep.to(hidden_states.dtype)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)
        additional_kwargs["encoder_seq_len"] = encoder_hidden_states.shape[1]

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, hidden_states)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states)
        )
        if base_ratio is not None:
            image_rotary_emb_base = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)
        else:
            image_rotary_emb_base = None
        image_rotary_emb = self.pos_embed(img_shapes, regional_txt_seq_lens, device=hidden_states.device)

        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    encoder_hidden_states_mask,
                    temb,
                    image_rotary_emb,
                )

            else:
                encoder_hidden_states, hidden_states, encoder_hidden_states_base = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    encoder_hidden_states_base=encoder_hidden_states_base, # added
                    encoder_hidden_states_base_mask=encoder_hidden_states_base_mask, # added
                    base_ratio=base_ratio,  # added for regional control
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    image_rotary_emb_base=image_rotary_emb_base, # added
                    # 根据当前 block 的索引是否满足特定条件，决定是否保留 additional_kwargs 中的 'regional_attention_mask' 字段。如果不满足，则从传入的参数中删除该字段，从而控制是否启用“区域注意力”（region control）。
                    joint_attention_kwargs=additional_kwargs if index_block % attention_kwargs['double_inject_blocks_interval'] == 0 else {k: v for k, v in additional_kwargs.items() if k != 'regional_attention_mask'}, # delete attention mask to avoid region control
                )

        # Use only the image part (hidden_states) from the dual-stream blocks
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
