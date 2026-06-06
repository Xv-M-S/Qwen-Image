from diffusers import DiffusionPipeline
import torch

model_name = "Qwen/Qwen-Image"

# Load the pipeline
if torch.cuda.is_available():
    torch_dtype = torch.bfloat16
    device = "cuda"
else:
    torch_dtype = torch.float32
    device = "cpu"

pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
pipe = pipe.to(device)

positive_magic = {
    "en": "Ultra HD, 4K, cinematic composition.", # for english prompt
    "zh": "超清，4K，电影级构图" # for chinese prompt
}

# Generate image
prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197".'''

prompt = ''' A coffee shop entrance features a chalkboard sign reading "知不可乎骤得,托遗响于悲风. 知不可乎骤得,托遗响于悲风.悟已往之不谏,知来者之可追。何事秋风悲画善，落叶聚还散，寒鸦栖复惊。相去日已远，衣带渐宽终不悔，为伊消得人憔悴。" with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful picture of a girl with a smile.'''
prompt = '''为传统风筝艺术展设计一张海报。背景使用蓝天白云的图片，并在中心放置一幅大型的传统风筝图案。在风筝下面有个标题是‘传统风筝的魅力’，字体要大且醒目；在标题的下面有一首小诗‘纸鸢翻古韵，春风载梦遥’。底部右下角放置活动时间‘大年初六’和地点‘港村’。整个画面要体现传统文化的韵味。'''
negative_prompt = " " # Recommended if you don't use a negative prompt.


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

image = pipe(
    prompt=prompt + positive_magic["en"],
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="cuda").manual_seed(42)
).images[0]

image.save("quick.png")