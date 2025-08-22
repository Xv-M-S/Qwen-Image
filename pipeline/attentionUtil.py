from pipeline.pipeline_qwenimage_regional import RegionalQwenImageAttnProcessor
from config.boxLossConfig import boxConfig


def register_attention_control(pipe, controller = None):
    """
    Register the attention control to the model.
    """
    attn_procs = {}
    cross_att_count = 0
    for name in pipe.transformer.attn_processors.keys():
        if 'transformer_blocks' in name and name.endswith("attn.processor"):
            attn_procs[name] = RegionalQwenImageAttnProcessor(attnstore=controller)
            cross_att_count += 1
        else:
            attn_procs[name] = pipe.transformer.attn_processors[name]
    pipe.transformer.set_attn_processor(attn_procs)

    if controller is not None:
        controller.num_att_layers = cross_att_count
    
    if boxConfig.train_layer is not None:
        controller.num_att_layers = len(boxConfig.train_layer)