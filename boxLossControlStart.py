from pipeline.pipeline_qwenimage_regional import RegionalQwenImagePipeline, RegionalQwenImageAttnProcessor
from pipeline.QwenImageTransformerForRegional import RegionalQwenImageTransformer
from pipeline.attentionUtil import register_attention_control
from pipeline.attentionControl import AttentionStore
import torch
from pyinstrument import Profiler
from util.tool import draw_masks_on_image, visualize_mask_pairs
import os
from config.boxLossConfig import boxConfig

ENABLE_PROFILER = False


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_size_gb(model, dtype=torch.float32):
    total_params = sum(p.numel() for p in model.parameters())
    bytes_per_param = 4 if dtype == torch.float32 else 2
    total_bytes = total_params * bytes_per_param
    total_mb = total_bytes / (1024**2)
    total_gb = total_mb / 1024
    return total_gb

def load_model():
    model_name = "Qwen/Qwen-Image"

    # Load the pipeline
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
        device = "cuda"
    else:
        torch_dtype = torch.float32
        device = "cpu"

    pipe = RegionalQwenImagePipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
    # change the transformer model to regional
    pipe.transformer = RegionalQwenImageTransformer.from_pretrained(model_name, subfolder="transformer", torch_dtype=torch_dtype)

    controller = AttentionStore()
    register_attention_control(pipe, controller=controller)
    pipe = pipe.to(device)

    # 启用梯度检查点 - 反向时重新计算中间值，节省内存空间
    pipe.transformer.enable_gradient_checkpointing()


    """
    VAE 显存占用 (FP32): 0.23635575734078884 GB
    Text Encoder 显存占用 (FP16): 15.445363998413086 GB
    Transformer 显存占用 (FP32): 38.05458748340607 GB
    """
    print("VAE 显存占用 (FP32):", model_size_gb(pipe.vae, torch.float16), "GB")
    print("Text Encoder 显存占用 (FP16):", model_size_gb(pipe.text_encoder, torch.float16), "GB")
    print("Transformer 显存占用 (FP32):", model_size_gb(pipe.transformer, torch.float16), "GB")

    # for name, param in pipe.transformer.named_parameters():
    #     print(name, param.shape)

    return pipe, controller

def get_hw():
    # Generate with different aspect ratios
    aspect_ratios = {
        "1:1": (1328, 1328),
        "16:9": (1664, 928),
        "9:16": (928, 1664),
        "4:3": (1472, 1104),
        "3:4": (1104, 1472),
        "3:2": (1584, 1056),
        "2:3": (1056, 1584),
    }

    width, height = aspect_ratios["16:9"]

    boxConfig.H = height
    boxConfig.W = width

    return height, width

def prepare_regional_control(height, width):
    ## regional prompt and mask settings
    from testCase import regional_prompt_mask_pairs2 as regional_prompt_mask_pairs
    from util.tool import get_child_boxes
    from util.textComposition import visualize_structured_boxes_with_text

    regional_prompt_mask_pairs = get_child_boxes(regional_prompt_mask_pairs, width, height)
    print(regional_prompt_mask_pairs)
    visualize_structured_boxes_with_text(width, height, regional_prompt_mask_pairs, output_path="./runing_output_tempfile/output_structured_boxes_with_text.png")

    regional_prompts = []
    regional_masks = []
    regional_boxes = []
    regional_child_boxes = []

    background_prompt = "a photo" # set by default, but if you want to enrich background, you can set it to a more descriptive prompt
    background_prompt = '''A coffee shop entrance, '''

    background_mask = torch.ones((height, width))
    for region_idx, region in regional_prompt_mask_pairs.items():
        description = region['description']
        mask = region['mask']
        child_mask = region['child_boxes']
        regional_boxes.append(mask)
        x1, y1, x2, y2 = mask
        mask = torch.zeros((height, width))
        mask[y1:y2, x1:x2] = 1.0
        background_mask -= mask
        regional_prompts.append(description)
        regional_masks.append(mask)
        regional_child_boxes.append(child_mask)

    # if regional masks don't cover the whole image, append background prompt and mask
    whole_regional_mask = torch.ones((height, width)) - background_mask
    if background_mask.sum() > 0:
        regional_prompts.append(background_prompt)
        regional_masks.append(background_mask)

    return regional_prompts, regional_masks, regional_boxes, whole_regional_mask, regional_prompt_mask_pairs, regional_child_boxes

def prepare_base_control():
    # base prompt settings
    base_prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup", and a neon light  displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197".  '''
    base_prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup" , and a neon light  displaying "通义千问" . A poster showing a beautiful Chinese woman , and "π≈3.1415926-53589793-23846264-33832795-02384197" is written on the wall .'''
    base_prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup" , and a neon light  displaying "通义千问" . A poster showing a beautiful Chinese woman. the text "庐山云雾" is written on the wall .'''
    base_prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup" , and a neon light  displaying "通义千问" . the text "庐山云雾" is written on the wall .'''
    base_prompt = '''the text "庐山云雾" is written on the wall .'''
    base_prompt = '''A coffee shop entrance features a chalkboard sign reading "咖啡店" , and a neon light  displaying "通义千问" . the text "π≈3.1415926" is written on the wall .'''
    # base_prompt = '''A coffee shop entrance, '''

    negative_prompt = " " # Recommended if you don't use a negative prompt.

    positive_magic = {
        "en": "Ultra HD, 4K, cinematic composition.", # for english prompt
        "zh": "超清，4K，电影级构图" # for chinese prompt
    }

    return base_prompt, negative_prompt, positive_magic

def run():
    ## region control factor settings [超参数]
    mask_inject_steps = 0 # larger means stronger control, recommended between 5-10
    double_inject_blocks_interval = 1 # 1 means strongest control
    # single_inject_blocks_interval = 1 # 1 means strongest control
    base_ratio = 0.1 # smaller means stronger control
    save_path = "./runing_output_tempfile"


    pipe, controller = load_model()
    height,width = get_hw()
    base_prompt, negative_prompt, positive_magic = prepare_base_control()
    regional_prompts, regional_masks, regional_boxes, whole_regional_mask, regional_prompt_mask_pairs, regional_child_boxes = prepare_regional_control(height, width)

    ## visual layout
    visualize_mask_pairs(regional_prompt_mask_pairs, width, height, os.path.join(save_path, "visual_layout.png"))

    image = pipe(
        base_prompt=base_prompt + positive_magic["en"],
        attention_store=controller, # added for attention store
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=50,
        true_cfg_scale=4.0, # do not use CFG
        generator=torch.Generator(device="cuda").manual_seed(70),
        mask_inject_steps=mask_inject_steps, # inject mask
        attention_kwargs={
            "regional_prompts": regional_prompts,
            "regional_masks": regional_masks,
            "regional_boxes": regional_boxes,
            "regional_child_boxes": regional_child_boxes,
            "double_inject_blocks_interval": double_inject_blocks_interval,
            # "single_inject_blocks_interval": single_inject_blocks_interval,
            "base_ratio": base_ratio,
            "whole_regional_mask": whole_regional_mask,  # 是否在相乘的时候只在mask上进行
            "enable_whole_regional_mask": False
        },
        gaussian_smoothing_kwargs={
            "sigma": 0.5,
            "kernel_size": 3,
            "smooth_attentions": True
        }
    ).images[0]

    image_path = os.path.join(save_path, "example.png")
    image.save(image_path)

    # visual layout on image
    draw_masks_on_image(image_path, regional_prompt_mask_pairs, output_path=os.path.join(save_path, boxConfig.save_name))





if __name__ == "__main__":

    if ENABLE_PROFILER:
        profiler = Profiler()
        profiler.start()

    run()

    
    if ENABLE_PROFILER:
        profiler.stop()
        profiler.print()