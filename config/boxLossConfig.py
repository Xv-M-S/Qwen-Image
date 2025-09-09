from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple


@dataclass
class RunConfig:

    # image size
    W: int = 512
    H: int = 512
    now_step: int = 50
    text_len: int = 20

    # chose percent
    P: float = 0.01
    # number of pixels around the corner to be selected
    L: int = 1
    # threadhold keys
    thresholds: Dict[int, float] = field(default_factory=lambda: {0: 0.05, 10: 0.5, 20: 0.8})
    # refine
    refine: bool = True
    # update step scale
    scale_factor: int = 0.02
    scale_range: tuple = field(default_factory=lambda: (1.0, 0.5))
    scale_range_value: List[int] = field(default_factory=lambda: [
        1,1,1,1,1,
        0.1,0.1,0.1,0.1,0.1,
        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1
    ])
    bbox: List[List[int]] = field(default_factory=lambda: [[0, 0, 512, 512]])
    # move some model to cpu
    save_cpu_offload: bool = False

    ## new added for box loss
    # visual veature map
    visual_middle_res: bool = False
    visual_attention_map: bool = False
    # if to print cost time
    print_cost_time: bool = False
    # the position of the text
    text_index: Dict[str, List[int]] = field(default_factory=lambda: {"0":[0]})
    total_weight: float = 1.0

    # decided if to use global box loss gradient
    use_global_box_loss: bool = True
    latents_gaussian: bool = False  # 由于对初始变量经过超过80次的梯度更新，会导致latents失去语义，猜测由于其偏离了高斯分布


    # Number of denoising steps to apply attend-and-excite
    max_iter_to_alter: List[int] = field(default_factory=lambda: [
        0,# 1,2,3,4,# 5,6,7,8,9,
        # 10,11,12,13,14,15,16,17,18,19,
        # 20,21,22,23,24,25,26,27,28,29,
        # 30,31,32,33,34,35,36,37,38,39,
        # 40,41,42,43,44,45,46,47,48,49
    ])

    # max refinement steps
    max_refinement_steps: Dict[int, int] = field(default_factory=lambda: {0:80,1:8,2:4,3:2,4:1,5:16,6:16,7:16,8:16,9:16})
    default_value = 0

    # save name
    save_name: str = "example_with_mask_continue_5_refine_1_rnb_7.png"

    # train_layer
    train_layer: set = field(default_factory=lambda: {
        "0","1","2","3","4","5","6","7","8","9",
        "10","11","12","13","14","15","16","17","18","19",
        "20","21","22","23","24","25","26","27","28","29",
        "30","31","32","33","34","35","36","37","38","39",
        "40","41","42","43","44","45","46","47","48","49",
        "50","51","52","53","54","55","56","57","58",
        # "59" # 最后一层似乎触犯了天条，只要加上就会超内存。。。
    })

    # which feature map to use
    # feature_map: set = field(default_factory=lambda:{"img-to-txt","txt-to-img"})
    feature_map: set = field(default_factory=lambda:{"txt-to-img"})

    # switch
    switch_box_loss: bool = True

    # loss type
    # lossType = "diff" # "rnb"
    lossType = "rnb"

boxConfig = RunConfig()
