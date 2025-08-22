import abc
import torch
from config.boxLossConfig import boxConfig

class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    # @property
    # def num_uncond_att_layers(self):
    #     return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            self.forward(attn, is_cross)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"cross": [], "self": []}

    def forward(self, attn, is_cross: bool):
        key = f"{'cross' if is_cross else 'self'}"
        # for mmdit architecture, we can just use special layer of attention
        self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        self.attention_store = self.step_store
        # add every step map value to global store
        if self.save_global_store:
            with torch.no_grad():
                if len(self.global_store) == 0:
                    self.global_store = self.step_store
                else:
                    for key in self.global_store:
                        for i in range(len(self.global_store[key])):
                            self.global_store[key][i] += self.step_store[key][i].detach()
        self.step_store = self.get_empty_store()

    def get_store_attention(self):
        return self.attention_store

    def get_average_global_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.global_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}
    
    def clear_attention_store(self):
        self.attention_store = {}

    def __init__(self, save_global_store=False):
        '''
        Initialize an empty AttentionStore
        :param step_index: used to visualize only a specific step in the diffusion process
        '''
        super(AttentionStore, self).__init__()
        self.save_global_store = save_global_store
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}
        self.curr_step_index = 0
        self.num_uncond_att_layers = 0