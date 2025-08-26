from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple


@dataclass
class RunConfig:
    W: int = 512
    H: int = 512
    now_step: int = 50
    text_len: int = 20

    visual_middle_res: bool = False
    visual_attention_map: bool = False

    print_cost_time: bool = False

    text_index: Dict[str, List[int]] = field(default_factory=lambda: {"0":[0]})
    bbox: List[List[int]] = field(default_factory=lambda: [[0, 0, 512, 512]])

    P: float = 0.2
    # number of pixels around the corner to be selected
    L: int = 1
    # threadhold keys
    thresholds: Dict[int, float] = field(default_factory=lambda: {0: 0.05, 10: 0.5, 20: 0.8})
    # refine
    refine: bool = True
    # update step scale
    scale_factor: int = 0.01
    scale_range: tuple = field(default_factory=lambda: (1.0, 0.5))

    # Number of denoising steps to apply attend-and-excite
    max_iter_to_alter: int = 20

    # max refinement steps
    max_refinement_steps = 35

    # move some model to cpu
    save_cpu_offload: bool = False

    # train_layer
    train_layer: set = field(default_factory=lambda: {
        # "0","1","2","3","4","5","6","7","8","9",
        # "10","11","12","13","14","15","16","17","18","19",
        # "20","21","22","23","24","25","26","27","28","29",
        "30","31","32","33","34","35","36","37","38","39",
        # "40","41","42","43","44","45","46","47","48","49",
        # "50","51","52","53","54","55","56","57","58","59",
        # "60","61","62","63","64","65","66","67","68","69"
    })

    # switch
    switch_box_loss: bool = True

boxConfig = RunConfig()
