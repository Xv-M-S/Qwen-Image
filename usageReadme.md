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

# for read source code 

``` bash
git clone git@github.com:huggingface/diffusers.git
cd diffusers
pip install -e .
```


# download model 

# quick start

``` bash
python quickStart.py
```

# use demo to start

``` bash
pip install gradio
pip install dashscope
export NUM_GPUS_TO_USE=1          # Number of GPUs to use
export TASK_QUEUE_SIZE=100        # Task queue size
export TASK_TIMEOUT=300           # Task timeout in seconds
export DASHSCOPE_API_KEY=sk-3179f054c4a149e4ad40aadd2c5b6755
python examples/demo.py
```

# test hyperparameter
mainly test inject_steps and base_ration parameters 
``` bash
python hyperParamExp.py
```
