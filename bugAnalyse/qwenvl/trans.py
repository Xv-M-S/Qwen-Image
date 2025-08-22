from transformers import AutoTokenizer

# 加载 Qwen2.5-VL 的 tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True)

# 定义模板
template = '<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n'

# 假设用户输入为空（只看模板）
prompt = template.format("hello world!")

# Tokenize
tokens = tokenizer.tokenize(prompt)
print(f"Total tokens in template: {len(tokens)}")

# 查看前 40 个 tokens
print("First 40 tokens:")
for i, token in enumerate(tokens[:40]):
    print(f"{i}: {repr(token)}")