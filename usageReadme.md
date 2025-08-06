# download environment

``` bash
conda create -n qwenImageEnv python=3.10
conda activate qwenImageEnv
# 安装最新的 diffusers
pip install git+https://github.com/huggingface/diffusers

# 安装的transformers>=4.51.3 
pip install transformers==4.55.0

# 安装其他安装包
pip install -r requirements.txt

pip install accelerate
```

# quick start

``` bash
python quickStart.py
```