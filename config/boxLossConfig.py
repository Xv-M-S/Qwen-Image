from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple


@dataclass
class RunConfig:
    P: float = 0.2
    # number of pixels around the corner to be selected
    L: int = 1
    # threadhold keys
    thresholds: Dict[int, float] = field(default_factory=lambda: {0: 0.05, 10: 0.5, 20: 0.8})
    # refine
    refine: bool = True
    # update step scale
    scale_factor: int = 20
    scale_range: tuple = field(default_factory=lambda: (1.0, 0.5))

    # Number of denoising steps to apply attend-and-excite
    max_iter_to_alter: int = 10

    # max refinement steps
    max_refinement_steps = 6

    # move some model to cpu
    save_cpu_offload: bool = False

    # train_layer
    train_layer: set = field(default_factory=lambda: {"0","1","2","3"})

    # switch
    switch_box_loss: bool = True

boxConfig = RunConfig()
