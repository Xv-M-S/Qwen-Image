from pipeline.pipeline_qwenimage_regional import RegionalQwenImagePipeline, RegionalQwenImageAttnProcessor
from pipeline.QwenImageTransformerForRegional import RegionalQwenImageTransformer
import torch
from pipeline.attentionUtil import register_attention_control

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

register_attention_control(pipe, controller=None)

pipe = pipe.to(device)

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


## regional prompt and mask settings
regional_prompt_mask_pairs = {
    "0": {
        "description": ''' a chalkboard sign reading "Qwen Coffee 😊 $2 per cup" ''',
        "mask": [128, 128, 384, 640]
    },
    "1": {
        "description": ''' a plaque sign "通义千问" ''',
        "mask": [500, 48, 756, 112]
    },
    "2": {
        "description": ''' a poster is written "π≈3.1415926-53589793-23846264-33832795-02384197" ''',
        "mask": [500, 640, 756, 780]
    }
}

regional_prompts = []
regional_masks = []
background_prompt = "a photo" # set by default, but if you want to enrich background, you can set it to a more descriptive prompt
background_prompt = '''A coffee shop entrance, '''
background_mask = torch.ones((height, width))
for region_idx, region in regional_prompt_mask_pairs.items():
    description = region['description']
    mask = region['mask']
    x1, y1, x2, y2 = mask
    mask = torch.zeros((height, width))
    mask[y1:y2, x1:x2] = 1.0
    background_mask -= mask
    regional_prompts.append(description)
    regional_masks.append(mask)

# if regional masks don't cover the whole image, append background prompt and mask
whole_regional_mask = torch.ones((height, width)) - background_mask
if background_mask.sum() > 0:
    regional_prompts.append(background_prompt)
    regional_masks.append(background_mask)

## region control factor settings
mask_inject_steps = 50 # larger means stronger control, recommended between 5-10
double_inject_blocks_interval = 1 # 1 means strongest control
# single_inject_blocks_interval = 1 # 1 means strongest control
base_ratio = 0.1 # smaller means stronger control

# base prompt settings
base_prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," and a neon light  displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197".  '''
base_prompt = '''A coffee shop entrance, '''

negative_prompt = " " # Recommended if you don't use a negative prompt.

positive_magic = {
    "en": "Ultra HD, 4K, cinematic composition.", # for english prompt
    "zh": "超清，4K，电影级构图" # for chinese prompt
}


image = pipe(
    base_prompt=base_prompt + positive_magic["en"],
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
        "double_inject_blocks_interval": double_inject_blocks_interval,
        # "single_inject_blocks_interval": single_inject_blocks_interval,
        "base_ratio": base_ratio,
        "whole_regional_mask": whole_regional_mask,  # 是否在相乘的时候只在mask上进行
        "enable_whole_regional_mask": False
    }
).images[0]

image.save("example.png")