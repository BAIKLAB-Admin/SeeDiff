from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
from diffusers import StableDiffusionPipeline
import torch.nn.functional as nnf
import numpy as np
import abc
import ptp_utils
import seq_aligner
import cv2
import json
import argparse
import multiprocessing as mp
import threading
from random import choice
import os
import argparse
from IPython.display import Image, display
from pamr import PAMR
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt
import gradio as gr


LOW_RESOURCE = False 
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 5
MAX_NUM_WORDS = 77

coco_category_list_check_person = [    
    "arm",
    'person',
    "man",
    "woman",
    "child",
    "boy",
    "girl",
    "teenager"
]


VOC_category_list_check = {
    'aeroplane':['aerop','lane'],
    'bicycle':['bicycle'],
    'bird':['bird'],
    'boat':['boat'],
    'bottle':['bottle'],
    'bus':['bus'],
    'car':['car'],
    'cat':['cat'],
    'chair':['chair'],
    'cow':['cow'],
    'diningtable':['table'],
    'dog':['dog'],
    'horse':['horse'],
    'motorbike':['motorbike'],
    'person':coco_category_list_check_person,
    'pottedplant':['pot','plant','ted'],
    'sheep':['sheep'],
    'sofa':['sofa'],
    'train':['train'],
    'tvmonitor':['monitor','tv','monitor']
    }


coco_category_list_check = [    "arm",'aerop','lane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'table',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pot',
    'ted',
    'plant',
    'sheep',
    'sofa',
    'train',
    'tv',
    'monitor']

coco_category_to_id_v1 = { 'aeroplane':0,
    'bicycle':1,
    'bird':2,
    'boat':3,
    'bottle':4,
    'bus':5,
    'car':6,
    'cat':7,
    'chair':8,
    'cow':9,
    'diningtable':10,
    'dog':11,
    'horse':12,
    'motorbike':13,
    'person':14,
    'pottedplant':15,
    'sheep':16,
    'sofa':17,
    'train':18,
    'tvmonitor':19}


coco_category_list = [ 
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor']

classes = {
    0: 'background',
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor',
    21: 'rider'
}



class LocalBlend:

    def __call__(self, x_t, attention_store):
        k = 1
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
        maps = torch.cat(maps, dim=1)
        maps = (maps * self.alpha_layers).sum(-1).mean(1)
        mask = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(mask, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.threshold)
        mask = (mask[:1] + mask[1:]).float()
        x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t
       
    def __init__(self, prompts: List[str], words: [List[List[str]]], threshold=.3,tokenizer=None,device=None):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.to(device)
        self.threshold = threshold


class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class EmptyControl(AttentionControl):
    
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        return attn
    
    
class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
#         if attn.shape[1] <= 128 ** 2:  # avoid memory overhead
        self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention
    

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]],tokenizer=None):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch.float32)
#     print(values)
    for word in word_select:
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = values
    return equalizer



from PIL import Image

def load_sa_map(load_dir , res , image_cnt , device):
    sa_img = Image.open(os.path.join(load_dir , f'sa_{res}_{image_cnt}.png'))
    sa_tensor = torch.tensor(np.array(sa_img)).to(device)
    sa_tensor_norm = (sa_tensor - sa_tensor.min()) / (sa_tensor.max() - sa_tensor.min())
    return sa_tensor_norm


def aggregate_attention_light(attn_save_dir = None , class_one = None , image_cnt = None , device=None):
    base_load_dir = os.path.join(attn_save_dir , class_one)
    ca_16_init_map = torch.load(os.path.join(base_load_dir , f'ca_16_{image_cnt}.pth'))
    
    sa_16_map = load_sa_map(base_load_dir , 16 , image_cnt , device)
    sa_32_map = load_sa_map(base_load_dir , 32 , image_cnt , device)
    sa_64_map = load_sa_map(base_load_dir , 64 , image_cnt , device)
    
    
    ca_map = torch.sum(ca_16_init_map , dim=0)
    ca_map_norm = (ca_map - ca_map.min()) / (ca_map.max() - ca_map.min())
    ca_map_reshape = ca_map_norm.reshape(16*16)
    # Otsu 를 적용한 dynamic threshold 
    ca_int_map = (ca_map_reshape*ca_map_reshape).reshape(16,16)
    ca_int_map = np.array((ca_int_map*255).to('cpu')).astype(np.uint16)
    ret, ca_map_mask = cv2.threshold(ca_int_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ca_map_mask = torch.tensor(ca_map_mask.astype(np.uint8)).reshape(16*16)
    ca_map_mask = ca_map_mask > 0    
    top_idx = torch.tensor(torch.where(ca_map_mask == True)[0])
    cos = torch.nn.CosineSimilarity(dim=1)

    vis_map = torch.zeros(sa_16_map.shape[0])
    for idx in top_idx:
        anchor_token = sa_16_map[idx]
        sim_embedding = cos(sa_16_map , anchor_token.unsqueeze(0)) 
        sim_embedding = sim_embedding
        vis_map = vis_map + sim_embedding.detach().to('cpu')
    
    sa_agg_16 = np.array(vis_map.reshape(16,16).to('cpu'))
    sa_agg_16_norm = (sa_agg_16 - sa_agg_16.min()) / (sa_agg_16.max() - sa_agg_16.min())


    # 32
    ca_init_map = torch.tensor(sa_agg_16_norm).to(device)
    ca_init_map = torch.tensor(cv2.resize(ca_init_map.to('cpu').numpy(), (32, 32), interpolation=cv2.INTER_CUBIC))
    # Otsu Dynamic Thresholding
    ca_int_map = (ca_init_map*ca_init_map).reshape(32,32)
    ca_int_map = np.array((ca_int_map*255).to('cpu')).astype(np.uint16)
    ret, ca_map_mask = cv2.threshold(ca_int_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ca_map_mask = torch.tensor(ca_map_mask.astype(np.uint8)).reshape(32*32)
    ca_map_mask = ca_map_mask > 0
    ca_map_mask = 1 - ca_map_mask*1
    top_idx = torch.tensor(torch.where(ca_map_mask == True)[0])
    vis_map = torch.zeros(sa_32_map.shape[0])
    for idx in top_idx:
        anchor_token = sa_32_map[idx]
        sim_embedding = cos(sa_32_map , anchor_token.unsqueeze(0)) 
        sim_embedding = sim_embedding
        vis_map = vis_map + sim_embedding.detach().to('cpu')
    
    sa_agg_32 = np.array(vis_map.reshape(32,32).to('cpu'))
    sa_agg_32_norm = (sa_agg_32 - sa_agg_32.min()) / (sa_agg_32.max() - sa_agg_32.min())
    sa_agg_32_norm = 1 - sa_agg_32_norm
    ca_init_map_norm = (ca_init_map - ca_init_map.min()) / (ca_init_map.max() - ca_init_map.min())
    sa_agg_32_norm = sa_agg_32_norm * np.array(ca_init_map_norm.to('cpu'))
    sa_agg_32_norm = (sa_agg_32_norm - sa_agg_32_norm.min()) / (sa_agg_32_norm.max() - sa_agg_32_norm.min())
    
    # 64
    ca_init_map = torch.tensor(sa_agg_32_norm).to(device)
    ca_init_map = torch.tensor(cv2.resize(ca_init_map.to('cpu').numpy(), (64, 64), interpolation=cv2.INTER_CUBIC))
    # Otsu Dynamic Thresholding
    ca_int_map = (ca_init_map*ca_init_map).reshape(64,64)
    ca_int_map = np.array((ca_int_map*255).to('cpu')).astype(np.uint16)
    ret, ca_map_mask = cv2.threshold(ca_int_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ca_map_mask = torch.tensor(ca_map_mask.astype(np.uint8)).reshape(64*64)
    ca_map_mask = ca_map_mask > 0
    ca_map_mask = 1 - ca_map_mask*1
    top_idx = torch.tensor(torch.where(ca_map_mask == True)[0])
    vis_map = torch.zeros(sa_64_map.shape[0])
    for idx in top_idx:
        anchor_token = sa_64_map[idx]
        sim_embedding = cos(sa_64_map , anchor_token.unsqueeze(0)) 
        sim_embedding = sim_embedding
        vis_map = vis_map + sim_embedding.detach().to('cpu')
    
    sa_agg_64 = np.array(vis_map.reshape(64,64).to('cpu'))
    sa_agg_64_norm = (sa_agg_64 - sa_agg_64.min()) / (sa_agg_64.max() - sa_agg_64.min())
    sa_agg_64_norm = 1 - sa_agg_64_norm
    ca_init_map_norm = (ca_init_map - ca_init_map.min()) / (ca_init_map.max() - ca_init_map.min())
    sa_agg_64_norm = sa_agg_64_norm * np.array(ca_init_map_norm.to('cpu'))
    sa_agg_64_norm = (sa_agg_64_norm - sa_agg_64_norm.min()) / (sa_agg_64_norm.max() - sa_agg_64_norm.min())
    res = sa_agg_64_norm 
    res = torch.tensor(cv2.resize(np.array(res), (512, 512), interpolation=cv2.INTER_CUBIC))
    return res

    
    
    






def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int, prompts=None , mask=None , ca_init_map = None , class_one = None , target_location = None , image_cnt = None , save_dir = None):
    out = []
    out_cross_norm = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    
    target_location = target_location
    if class_one == 'pottedplant' or class_one == 'tvmonitor' or class_one == 'cell phone' or class_one == 'parking meter':
        for location in from_where:
            for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                if item.shape[1] == num_pixels:
                    cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                    cross_maps_norm = torch.sum(cross_maps[:,:,:,target_location] , dim=0)
                    if cross_maps_norm.var() > 1e-4:
                        cross_maps_norm = (cross_maps_norm - cross_maps_norm.min()) / (cross_maps_norm.max() - cross_maps_norm.min())
                        out.append(cross_maps)
                        out_cross_norm.append(cross_maps_norm.unsqueeze(0))
                    else:
                        cross_maps_norm = (cross_maps_norm - cross_maps_norm.min()) / (cross_maps_norm.max() - cross_maps_norm.min())
                        cross_maps_norm = cross_maps_norm*0.01
                        out.append(cross_maps)
                        out_cross_norm.append(cross_maps_norm.unsqueeze(0))

        target_location = target_location + 2
        for location in from_where:
            for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                if item.shape[1] == num_pixels:
                    cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                    cross_maps_norm = torch.sum(cross_maps[:,:,:,target_location] , dim=0)
                    if cross_maps_norm.var() > 1e-4:
                        cross_maps_norm = (cross_maps_norm - cross_maps_norm.min()) / (cross_maps_norm.max() - cross_maps_norm.min())
                        out.append(cross_maps)
                        out_cross_norm.append(cross_maps_norm.unsqueeze(0))
                    else:
                        cross_maps_norm = (cross_maps_norm - cross_maps_norm.min()) / (cross_maps_norm.max() - cross_maps_norm.min())
                        cross_maps_norm = cross_maps_norm*0.01
                        out.append(cross_maps)
                        out_cross_norm.append(cross_maps_norm.unsqueeze(0))
    elif class_one == 'diningtable':
        target_location = target_location + 1
        for location in from_where:
            for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                if item.shape[1] == num_pixels:
                    cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                    cross_maps_norm = torch.sum(cross_maps[:,:,:,target_location] , dim=0)
                    if cross_maps_norm.var() > 1e-4:
                        cross_maps_norm = (cross_maps_norm - cross_maps_norm.min()) / (cross_maps_norm.max() - cross_maps_norm.min())
                        out.append(cross_maps)
                        out_cross_norm.append(cross_maps_norm.unsqueeze(0))
                    else:
                        cross_maps_norm = (cross_maps_norm - cross_maps_norm.min()) / (cross_maps_norm.max() - cross_maps_norm.min())
                        cross_maps_norm = cross_maps_norm*0.01
                        out.append(cross_maps)
                        out_cross_norm.append(cross_maps_norm.unsqueeze(0))
    else:
        for location in from_where:
            for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                if item.shape[1] == num_pixels:
                    cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                    cross_maps_norm = torch.sum(cross_maps[:,:,:,target_location] , dim=0)
                    if cross_maps_norm.var() > 1e-4:
                        cross_maps_norm = (cross_maps_norm - cross_maps_norm.min()) / (cross_maps_norm.max() - cross_maps_norm.min())
                        out.append(cross_maps)
                        out_cross_norm.append(cross_maps_norm.unsqueeze(0))
                    else:
                        cross_maps_norm = (cross_maps_norm - cross_maps_norm.min()) / (cross_maps_norm.max() - cross_maps_norm.min())
                        cross_maps_norm = cross_maps_norm*0.01
                        out.append(cross_maps)
                        out_cross_norm.append(cross_maps_norm.unsqueeze(0))


    out = torch.cat(out, dim=0)
    out_cross_norm = torch.cat(out_cross_norm, dim=0)

    

    # Cross-attention coordinates
    if mask is None:
        if res == 64:
            # res_64 origin code
            ca_map = torch.sum(out_cross_norm , dim=0)
            ca_map_norm = (ca_map - ca_map.min()) / (ca_map.max() - ca_map.min())
            # ca_map_reshape = ca_map_norm.reshape(num_pixels)

            # Chain Aggregation
            # ca_map = torch.sum(out_cross_norm , dim=0)
            # ca_map_norm = (ca_map - ca_map.min()) / (ca_map.max() - ca_map.min())


            # Otsu Dynamic Threshold
            ca_init_map = torch.tensor(ca_init_map).to(out_cross_norm.device)
            ca_int_map = (ca_init_map*ca_init_map).reshape(res,res)
            ca_int_map = np.array((ca_int_map*255).to('cpu')).astype(np.uint16)
            ret, ca_map_mask = cv2.threshold(ca_int_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # ca_map_mask = ca_init_map > 0.2
            ca_map_mask = torch.tensor(ca_map_mask.astype(np.uint8)).reshape(res*res)
            ca_map_mask = ca_map_mask > 0
            ca_map_mask = 1 - ca_map_mask*1
            top_idx = torch.tensor(torch.where(ca_map_mask == True)[0]) 

            # Origin Code
            # ca_init_map = torch.tensor(ca_init_map).to(out_cross_norm.device)
            # ca_map_reshape = ca_init_map.reshape(num_pixels)
            # ca_map_mask = ca_map_reshape > 0.4
            # ca_map_mask = 1 - ca_map_mask*1
            # heatmap 용 지워도됨
            
            # colormap = plt.get_cmap('viridis')  
            # heatmap_image = colormap(np.array(ca_map_norm.to('cpu')))
            # heatmap_image = (heatmap_image[:, :, :3] * 255).astype(np.uint8)
            # cv2.imwrite(f'Stable_Diffusion/for_figure/{res}/ca_init_heatmap_image.png', cv2.cvtColor(heatmap_image , cv2.COLOR_BGR2RGB))

        elif res==32:
            # Chain aggregation -> 기존 cross attn map + 이전 단계의 init map 활용
            ca_map = torch.sum(out_cross_norm , dim=0)
            ca_map_norm = (ca_map - ca_map.min()) / (ca_map.max() - ca_map.min())
            # ca_init_map = ca_init_map * 0.5 + ca_map_norm * 0.5
            
            ca_init_map = torch.tensor(ca_init_map).to(out_cross_norm.device)
            
            # Otsu Dynamic Thresholding
            ca_int_map = (ca_init_map*ca_init_map).reshape(res,res)
            ca_int_map = np.array((ca_int_map*255).to('cpu')).astype(np.uint16)
            ret, ca_map_mask = cv2.threshold(ca_int_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ca_map_mask = torch.tensor(ca_map_mask.astype(np.uint8)).reshape(res*res)
            ca_map_mask = ca_map_mask > 0
            ca_map_mask = 1 - ca_map_mask*1
            top_idx = torch.tensor(torch.where(ca_map_mask == True)[0])
            # Origin Code
            # ca_map_reshape = ca_init_map.reshape(num_pixels)
            # ca_map_mask = ca_map_reshape > 0.4
            # ca_map_mask = 1 - ca_map_mask*1
            # top_idx = torch.tensor(torch.where(ca_map_mask == True)[0])

            # cross attention guided
            ca_map = torch.sum(out_cross_norm , dim=0)
            ca_map_norm = (ca_map - ca_map.min()) / (ca_map.max() - ca_map.min())
            ca_map_reshape = ca_map_norm.reshape(num_pixels)
            ca_map_mask = ca_map_reshape > 0.4
            top_idx_cg = torch.tensor(np.where(ca_map_mask.to('cpu') == True)[0])
            # Cascade
            ca_init_map = torch.tensor(ca_init_map).to(out_cross_norm.device)
            ca_map_reshape = ca_init_map.reshape(num_pixels)
            ca_map_mask = ca_map_reshape > 0.4
            top_idx_cascade = torch.tensor(torch.where(ca_map_mask == True)[0])

        else:
            ca_map = torch.sum(out_cross_norm , dim=0)
            ca_map_norm = (ca_map - ca_map.min()) / (ca_map.max() - ca_map.min())
            ca_map_reshape = ca_map_norm.reshape(num_pixels)
            
            
            # Origin Code
            ca_map_mask = ca_map_reshape*ca_map_reshape > 0.5

            # Otsu 를 적용한 dynamic threshold 
            ca_int_map = (ca_map_reshape*ca_map_reshape).reshape(res,res)
            ca_int_map = np.array((ca_int_map*255).to('cpu')).astype(np.uint16)
            ret, ca_map_mask = cv2.threshold(ca_int_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ca_map_mask = torch.tensor(ca_map_mask.astype(np.uint8)).reshape(res*res)
            ca_map_mask = ca_map_mask > 0
            # top_idx = torch.where(ca_map_mask)[0]
            # eroded_mask = cv2.erode(np.array(ca_map_mask.detach().to('cpu')).astype(np.uint8)*255 , np.ones((2,2), np.uint8))
            # eroded_mask = np.array(eroded_mask == 255)

            # colormap = plt.get_cmap('viridis')  
            # heatmap_image = colormap(np.array(ca_map_norm.to('cpu')*ca_map_norm.to('cpu')))
            # heatmap_image = (heatmap_image[:, :, :3] * 255).astype(np.uint8)
            # cv2.imwrite(f'Stable_Diffusion/for_figure/{res}/ca_agg_heatmap_image.png', cv2.cvtColor(heatmap_image , cv2.COLOR_BGR2RGB))
            # cv2.imwrite(f'Stable_Diffusion/for_figure/{res}/thresholded_ca_agg_heatmap_image.png', np.array(ca_map_mask.to('cpu').reshape(res,res))*255)
            
            top_cg_idx = torch.tensor(np.where(ca_map_mask.to('cpu') == True)[0])
            
            
            top_idx = torch.tensor(torch.where(ca_map_mask == True)[0])
            del ca_map_mask , ca_map

    else:
        ca_map = torch.sum(out_cross_norm , dim=0)
        ca_map_norm = (ca_map - ca_map.min()) / (ca_map.max() - ca_map.min())
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        ca_map_norm = (ca_map_norm + mask.to(ca_map_norm.device))/2.0

        ca_map_reshape = ca_map_norm.reshape(num_pixels)
        ca_map_mask = ca_map_reshape > 1.5*ca_map_reshape.mean()
        # top_idx = torch.where(ca_map_mask)[0]
        


        
        
        eroded_mask = cv2.erode(np.array(ca_map_mask.detach().to('cpu')).astype(np.uint8)*255 , np.ones((2,2), np.uint8))
        eroded_mask = np.array(eroded_mask == 255)
        top_idx = torch.tensor(np.where(eroded_mask == True)[0])
        del ca_map_mask , ca_map

        


    # top_idx = torch.where(ca_map_reshape > ca_map_reshape.mean())[0]

    # Self-attention aggregation
    out_self = []
    for location in from_where:
        for item in attention_maps[f"{location}_{'self' if is_cross else 'cross'}"]:
            if item.shape[1] == num_pixels:
                self_maps = torch.sum(item , dim=0)
                out_self.append(self_maps)

    sa_out = out_self[0]
    
    # target_sa = np.array((sa_out * 255).to('cpu')).astype(np.uint8)
    # sa_image = Image.fromarray(target_sa)
    # sa_image.save(os.path.join(class_save_dir , f'sa_{res}_{image_cnt}.png'), 'PNG')

    cos = torch.nn.CosineSimilarity(dim=1)
    vis_map = torch.zeros(out_self[0].shape[0])
    for idx in top_idx:
        anchor_token = out_self[0][idx]
        sim_embedding = cos(out_self[0] , anchor_token.unsqueeze(0)) 
        sim_embedding = sim_embedding
        vis_map = vis_map + sim_embedding.detach().to('cpu')
        # origin
        # weighted = 2 * ca_map_reshape[idx] 
        
        
        # sim_embedding = vis_tokens @ anchor_token

        # sim_score = cos(cur_embedding , last_embedding)
        # if ((sim_embedding / sim_embedding.max()).var() < var_threshold):
        # sim_embedding = (sim_embedding - sim_embedding.min()) / (sim_embedding.max() - sim_embedding.min())
        # if res > 15:
        #     a = np.array(sim_embedding.reshape(res,res).to('cpu'))
        #     a = (a - a.min()) / (a.max() - a.min())
            # colormap = plt.get_cmap('viridis')  # 여기서 'viridis' 대신 다른 colormap을 사용할 수 있음
            # heatmap_image = colormap(a)

            # # # heatmap_image는 RGBA 형식이므로 RGB로 변환 (알파 채널 제거)
            # heatmap_image = (heatmap_image[:, :, :3] * 255).astype(np.uint8)
            # k = idx.item()
            # # 히트맵 이미지 저장
            # cv2.imwrite(f'Stable_Diffusion/for_figure/{res}/sa_heatmap_image_{k}.png', cv2.cvtColor(heatmap_image , cv2.COLOR_BGR2RGB))
        
        # origin
        # sim_embedding = sim_embedding * weighted

            # last_embedding = cur_embedding
    
    if res > 31:
        # vis_map_cg = torch.zeros(out_self[0].shape[0])
        # vis_map_cascade = torch.zeros(out_self[0].shape[0])

        # for idx in top_idx_cg:
        #     anchor_token = out_self[0][idx]
        #     sim_embedding = cos(out_self[0] , anchor_token.unsqueeze(0)) 
        #     sim_embedding = sim_embedding
        #     vis_map_cg = vis_map_cg + sim_embedding.detach().to('cpu')
        # for idx in top_idx_cascade:
        #     anchor_token = out_self[0][idx]
        #     sim_embedding = cos(out_self[0] , anchor_token.unsqueeze(0)) 
        #     sim_embedding = sim_embedding
        #     vis_map_cascade = vis_map_cascade + sim_embedding.detach().to('cpu')
   
   
        # vis_map = (vis_map - vis_map.min()) / (vis_map.max() - vis_map.min())
        # vis_map_cg = (vis_map_cg - vis_map_cg.min()) / (vis_map_cg.max() - vis_map_cg.min())
        # vis_map_cascade = (vis_map_cascade - vis_map_cascade.min()) / (vis_map_cascade.max() - vis_map_cascade.min())
        # vis_map_cg = vis_map_cg.reshape(res, res)
        # vis_map_cascade = vis_map_cascade.reshape(res,res)

        # heatmap_image = colormap(np.array(vis_map.reshape(res,res).to('cpu')))

        # #     # heatmap_image는 RGBA 형식이므로 RGB로 변환 (알파 채널 제거)
        # heatmap_image = (heatmap_image[:, :, :3] * 255).astype(np.uint8)

        #     # 히트맵 이미지 저장
        # cv2.imwrite(f'Stable_Diffusion/for_figure/{res}/sa_heatmap_image_{idx}.png', cv2.cvtColor(heatmap_image , cv2.COLOR_BGR2RGB))


        
        if res == 64 or res == 32:

            # cv2.imwrite(f'Stable_Diffusion/for_figure/{res}/reverse_map.png' , cv2.cvtColor(heatmap_image , cv2.COLOR_BGR2RGB))
            
            vis_map = 1 - vis_map
            vis_map = vis_map.reshape(res , res)
            vis_map = (vis_map - vis_map.min()) / (vis_map.max() - vis_map.min())
            ca_init_map_norm = (ca_init_map - ca_init_map.min()) / (ca_init_map.max() - ca_init_map.min())

            # heatmap_image = colormap(np.array(ca_init_map_norm.to('cpu')))
            # heatmap_image = (heatmap_image[:, :, :3] * 255).astype(np.uint8)
            # cv2.imwrite(f'Stable_Diffusion/for_figure/{res}/init_at_{res}.png' , cv2.cvtColor(heatmap_image , cv2.COLOR_BGR2RGB))
            vis_map = ca_init_map_norm.to('cpu') * vis_map
            vis_map = (vis_map - vis_map.min()) / (vis_map.max() - vis_map.min())
            # colormap = plt.get_cmap('viridis')
            # heatmap_image = colormap(vis_map)

            # #     # heatmap_image는 RGBA 형식이므로 RGB로 변환 (알파 채널 제거)
            # heatmap_image = (heatmap_image[:, :, :3] * 255).astype(np.uint8)
            # cv2.imwrite(f'Stable_Diffusion/for_figure/{res}/final_{res}.png' , cv2.cvtColor(heatmap_image , cv2.COLOR_BGR2RGB))
            
            

        if res ==32:
            # sa_out = torch.matrix_power(sa_out, 4)
            # sa_out = (sa_out @ ca_map.reshape(res **2 , 1)).reshape(res,res)
            # sa_out_norm = (sa_out - sa_out.min()) / (sa_out.max() - sa_out.min())

            return out.cpu() , vis_map.cpu() 
        
        elif res == 64:
            return out.cpu() , vis_map.cpu() 




    elif res == 16:
        # vis_map = ca_map_norm.reshape(res*res).to('cpu') * vis_map
        vis_map = ( vis_map - vis_map.min()) / (vis_map.max() - vis_map.min())
        
        # 지워야함.
        # vis_map = 1 - vis_map

        vis_map = vis_map.reshape(res , res)
        # colormap = plt.get_cmap('viridis')  # 여기서 'viridis' 대신 다른 colormap을 사용할 수 있음
        # heatmap_image = colormap(vis_map)

        #     # heatmap_image는 RGBA 형식이므로 RGB로 변환 (알파 채널 제거)
        # heatmap_image = (heatmap_image[:, :, :3] * 255).astype(np.uint8)
        # cv2.imwrite(f'Stable_Diffusion/for_figure/{res}/res_{res}_sa_agg_map.png' , cv2.cvtColor(heatmap_image , cv2.COLOR_BGR2RGB))
    else:
        vis_map = vis_map.reshape(res , res)

    return out.cpu() , vis_map.cpu()

from collections import OrderedDict

def cross_attention_aggregation(i = None , ca_init_map_list = None , attention_maps = None , class_one = None , from_where = None , num_pixels = None , res = None , is_cross = None , select = None , target_location = None):
    
    out = []
    out_cross_norm = []
    if i == 0:
        if ca_init_map_list is not None:
            ca_init_map = ca_init_map_list[i]
        else:
            ca_init_map = None
    else:                
        if ca_init_map_list is not None:
            ca_init_map = ca_init_map_list[i]
        else:
            ca_init_map = None
    
    if class_one == 'pottedplant' or class_one == 'tvmonitor' or class_one == 'cell phone' or class_one == 'parking meter':
        for location in from_where:
            for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                if item.shape[1] == num_pixels:
                    cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[select]
                    cross_maps_norm = torch.sum(cross_maps[:,:,:,target_location] , dim=0)
                    if cross_maps_norm.var() > 1e-4:
                        cross_maps_norm = (cross_maps_norm - cross_maps_norm.min()) / (cross_maps_norm.max() - cross_maps_norm.min())
                        out.append(cross_maps)
                        out_cross_norm.append(cross_maps_norm.unsqueeze(0))
                    else:
                        cross_maps_norm = (cross_maps_norm - cross_maps_norm.min()) / (cross_maps_norm.max() - cross_maps_norm.min())
                        cross_maps_norm = cross_maps_norm*0.01
                        out.append(cross_maps)
                        out_cross_norm.append(cross_maps_norm.unsqueeze(0))
        target_location = target_location + 2
        for location in from_where:
            for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                if item.shape[1] == num_pixels:
                    cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[select]
                    cross_maps_norm = torch.sum(cross_maps[:,:,:,target_location] , dim=0)
                    if cross_maps_norm.var() > 1e-4:
                        cross_maps_norm = (cross_maps_norm - cross_maps_norm.min()) / (cross_maps_norm.max() - cross_maps_norm.min())
                        out.append(cross_maps)
                        out_cross_norm.append(cross_maps_norm.unsqueeze(0))
                    else:
                        cross_maps_norm = (cross_maps_norm - cross_maps_norm.min()) / (cross_maps_norm.max() - cross_maps_norm.min())
                        cross_maps_norm = cross_maps_norm*0.01
                        out.append(cross_maps)
                        out_cross_norm.append(cross_maps_norm.unsqueeze(0))
    elif class_one == 'diningtable':
        target_location = target_location + 1
        for location in from_where:
            for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                if item.shape[1] == num_pixels:
                    cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[select]
                    cross_maps_norm = torch.sum(cross_maps[:,:,:,target_location] , dim=0)
                    if cross_maps_norm.var() > 1e-4:
                        cross_maps_norm = (cross_maps_norm - cross_maps_norm.min()) / (cross_maps_norm.max() - cross_maps_norm.min())
                        out.append(cross_maps)
                        out_cross_norm.append(cross_maps_norm.unsqueeze(0))
                    else:
                        cross_maps_norm = (cross_maps_norm - cross_maps_norm.min()) / (cross_maps_norm.max() - cross_maps_norm.min())
                        cross_maps_norm = cross_maps_norm*0.01
                        out.append(cross_maps)
                        out_cross_norm.append(cross_maps_norm.unsqueeze(0))
    else:
        for location in from_where:
            for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                if item.shape[1] == num_pixels:
                    cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[select]
                    cross_maps_norm = torch.sum(cross_maps[:,:,:,target_location] , dim=0)
                    if cross_maps_norm.var() > 1e-4:
                        cross_maps_norm = (cross_maps_norm - cross_maps_norm.min()) / (cross_maps_norm.max() - cross_maps_norm.min())
                        out.append(cross_maps)
                        out_cross_norm.append(cross_maps_norm.unsqueeze(0))
                    else:
                        cross_maps_norm = (cross_maps_norm - cross_maps_norm.min()) / (cross_maps_norm.max() - cross_maps_norm.min())
                        cross_maps_norm = cross_maps_norm*0.01
                        out.append(cross_maps)
                        out_cross_norm.append(cross_maps_norm.unsqueeze(0))
    return ca_init_map , out , out_cross_norm

def self_attention_aggregation(from_where = None , attention_maps = None , is_cross = None , num_pixels = None):
    out_self = []
    for location in from_where:
        for item in attention_maps[f"{location}_{'self' if is_cross else 'cross'}"]:
            if item.shape[1] == num_pixels:
                self_maps = torch.sum(item , dim=0)
                out_self.append(self_maps)
    return out_self


def seed_extraction(mask = None , res = None , out_cross_norm = None , ca_init_map = None , num_pixels = None):
    if mask is None:
        if res == 64:
            ca_map = torch.sum(out_cross_norm , dim=0)
            ca_map_norm = (ca_map - ca_map.min()) / (ca_map.max() - ca_map.min())
            ca_init_map = torch.tensor(ca_init_map).to(out_cross_norm.device)
            ca_int_map = (ca_init_map*ca_init_map).reshape(res,res)
            ca_int_map = np.array((ca_int_map*255).to('cpu')).astype(np.uint16)
            ca_map_mask = ca_init_map > 0.4
            ca_map_mask = torch.tensor(np.array(ca_map_mask.to('cpu')).astype(np.uint8)).reshape(res*res)
            ca_map_mask = ca_map_mask > 0
            ca_map_mask = 1 - ca_map_mask*1
            top_idx = torch.tensor(torch.where(ca_map_mask == True)[0]) 

        elif res==32:
            ca_map = torch.sum(out_cross_norm , dim=0)
            ca_map_norm = (ca_map - ca_map.min()) / (ca_map.max() - ca_map.min())
            ca_init_map = torch.tensor(ca_init_map).to(out_cross_norm.device)
            # Otsu Dynamic Thresholding
            ca_int_map = (ca_init_map*ca_init_map).reshape(res,res)
            ca_int_map = np.array((ca_int_map*255).to('cpu')).astype(np.uint16)
            # ret, ca_map_mask = cv2.threshold(ca_int_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ca_map_mask = ca_init_map > 0.4
            ca_map_mask = torch.tensor(np.array(ca_map_mask.to('cpu')).astype(np.uint8)).reshape(res*res)
            ca_map_mask = ca_map_mask > 0
            ca_map_mask = 1 - ca_map_mask*1
            top_idx = torch.tensor(torch.where(ca_map_mask == True)[0])
            ca_map = torch.sum(out_cross_norm , dim=0)
            ca_map_norm = (ca_map - ca_map.min()) / (ca_map.max() - ca_map.min())
            ca_map_reshape = ca_map_norm.reshape(num_pixels)
            ca_map_mask = ca_map_reshape > 0.4
            top_idx_cg = torch.tensor(np.where(ca_map_mask.to('cpu') == True)[0])

            ca_init_map = torch.tensor(ca_init_map).to(out_cross_norm.device)
            ca_map_reshape = ca_init_map.reshape(num_pixels)
            ca_map_mask = ca_map_reshape > 0.4
            top_idx_cascade = torch.tensor(torch.where(ca_map_mask == True)[0])

        else:
            ca_map = torch.sum(out_cross_norm , dim=0)
            ca_map_norm = (ca_map - ca_map.min()) / (ca_map.max() - ca_map.min())
            ca_map_reshape = ca_map_norm.reshape(num_pixels)
            # ca_map_mask = ca_map_reshape > 1.2*ca_map_reshape.mean()
            # ca_map_mask = ca_map_reshape > 2*ca_map_reshape.mean()
            
            # Origin Code
            ca_map_mask = ca_map_reshape*ca_map_reshape > 0.5

            # Otsu 를 적용한 dynamic threshold 
            ca_int_map = (ca_map_reshape*ca_map_reshape).reshape(res,res)
            ca_int_map = np.array((ca_int_map*255).to('cpu')).astype(np.uint16)
            ret, ca_map_mask = cv2.threshold(ca_int_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ca_map_mask = torch.tensor(ca_map_mask.astype(np.uint8)).reshape(res*res)
            ca_map_mask = ca_map_mask > 0
            top_idx = torch.tensor(torch.where(ca_map_mask == True)[0])
            del ca_map_mask , ca_map

    else:
        ca_map = torch.sum(out_cross_norm , dim=0)
        ca_map_norm = (ca_map - ca_map.min()) / (ca_map.max() - ca_map.min())
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        ca_map_norm = (ca_map_norm + mask.to(ca_map_norm.device))/2.0

        ca_map_reshape = ca_map_norm.reshape(num_pixels)
        ca_map_mask = ca_map_reshape > 1.5*ca_map_reshape.mean()
        eroded_mask = cv2.erode(np.array(ca_map_mask.detach().to('cpu')).astype(np.uint8)*255 , np.ones((2,2), np.uint8))
        eroded_mask = np.array(eroded_mask == 255)
        top_idx = torch.tensor(np.where(eroded_mask == True)[0])
        del ca_map_mask , ca_map
    return top_idx

def png_save(self_attn_map = None , res=None , image_cnt = None , png_save_dir = None):
    sa_map_norm = (self_attn_map - self_attn_map.min()) / (self_attn_map.max() - self_attn_map.min())
    Image.fromarray( (np.array(sa_map_norm.to('cpu'))*255).astype(np.uint8) ).save(f'{png_save_dir}' , format='PNG' , optimize=True)
    del sa_map_norm , self_attn_map
    torch.cuda.empty_cache()
    



def region_expansion(top_idx = None , out_self = None , cos = None , res = None , ca_init_map = None , vis_map = None):
    for idx in top_idx:
        anchor_token = out_self[0][idx]
        sim_embedding = cos(out_self[0] , anchor_token.unsqueeze(0)) 
        sim_embedding = sim_embedding
        vis_map = vis_map + sim_embedding.detach().to('cpu')
        
    if res > 31:
        vis_map = 1 - vis_map
        vis_map = vis_map.reshape(res , res)
        vis_map = (vis_map - vis_map.min()) / (vis_map.max() - vis_map.min())
        ca_init_map_norm = (ca_init_map - ca_init_map.min()) / (ca_init_map.max() - ca_init_map.min())
        vis_map = torch.tensor(ca_init_map_norm).to(vis_map.device) * vis_map
        vis_map = (vis_map - vis_map.min()) / (vis_map.max() - vis_map.min())
    elif res == 16:
        vis_map = ( vis_map - vis_map.min()) / (vis_map.max() - vis_map.min())
        vis_map = vis_map.reshape(res , res)
    else:
        vis_map = vis_map.reshape(res , res)

    return vis_map


def aggregate_attention_batch(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int, prompts=None , mask=None , ca_init_map_list = None , class_one = None , target_location = None , image_cnt = None , save_dir = None , state = None):
    batch_size = len(prompts)
    attention_maps = attention_store.get_average_attention()
    # attention_maps_1 = OrderedDict()
    # attention_maps_2 = OrderedDict()
    # attention_maps_encoder_1 = OrderedDict()
    # attention_maps_encoder_2 = OrderedDict()
    # attention_maps_decoder_1 = OrderedDict()
    # attention_maps_decoder_2 = OrderedDict()

    # for location in from_where:
        
        # Self-attention
        # for idx , item in enumerate(attention_maps[f'{location}_self']):
        #     if idx == 0: 
        #         attention_maps_1[f'{location}_self'] = []
        #         attention_maps_2[f'{location}_self'] = []
            # if idx == 0 and location == 'up':
            #     attention_maps_encoder_1[f'{location}_self'] = []
            #     attention_maps_encoder_2[f'{location}_self'] = []
            # if idx == 0 and location == 'down':
            #     attention_maps_decoder_1[f'{location}_self'] = []
            #     attention_maps_decoder_2[f'{location}_self'] = []
            
            # layer_num = item.shape[0]
            # layer_num = layer_num / 2
            # attention_maps_1[f'{location}_self'].append(item[:int(layer_num)])
            # attention_maps_2[f'{location}_self'].append(item[int(layer_num):])
            # if location == 'up':
            #     attention_maps_encoder_1[f'{location}_self'].append(item[:int(layer_num)])
            #     attention_maps_encoder_2[f'{location}_self'].append(item[:int(layer_num)])
            # if location == 'down':
            #     attention_maps_decoder_1[f'{location}_self'].append(item[:int(layer_num)])
            #     attention_maps_decoder_2[f'{location}_self'].append(item[:int(layer_num)])
        
        
        # Cross-attention
        # for idx , item in enumerate(attention_maps[f'{location}_cross']):
        #     if idx == 0: 
        #         attention_maps_1[f'{location}_cross'] = []
        #         attention_maps_2[f'{location}_cross'] = []
            # if idx == 0 and location == 'up':
            #     attention_maps_encoder_1[f'{location}_cross'] = []
            #     attention_maps_encoder_2[f'{location}_cross'] = []
            # if idx == 0 and location == 'down':
            #     attention_maps_decoder_1[f'{location}_cross'] = []
            #     attention_maps_decoder_2[f'{location}_cross'] = []
            # layer_num = item.shape[0]
            # layer_num = layer_num / 2
            # attention_maps_1[f'{location}_cross'].append(item[:int(layer_num)])
            # attention_maps_2[f'{location}_cross'].append(item[int(layer_num):])
            # if location == 'up':
            #     attention_maps_encoder_1[f'{location}_cross'].append(item[:int(layer_num)])
            #     attention_maps_encoder_2[f'{location}_cross'].append(item[:int(layer_num)])
            # if location == 'down':
            #     attention_maps_decoder_1[f'{location}_cross'].append(item[:int(layer_num)])
            #     attention_maps_decoder_2[f'{location}_cross'].append(item[:int(layer_num)])

    num_pixels = res ** 2
    target_location = target_location
    batch_size = len(prompts)
    out_list = []
    vis_map_list = []
    for i in range(batch_size):
        image_cnt = image_cnt + i
        ca_init_map , out , out_cross_norm = cross_attention_aggregation(i=i , ca_init_map_list = ca_init_map_list, attention_maps = attention_maps, class_one=class_one , from_where=from_where , num_pixels=num_pixels , res=res , is_cross = is_cross , select = select , target_location = target_location)    
        # if state == 'base':
        #     _ , _ , out_cross_norm_encoder = cross_attention_aggregation(i=i , ca_init_map_list = ca_init_map_list, attention_maps_1 = attention_maps_encoder_1 , attention_maps_2=attention_maps_encoder_2, class_one=class_one , from_where= ["up"] , num_pixels=num_pixels , res=res , is_cross = is_cross , select = select , target_location = target_location)
        #     _ , _ , out_cross_norm_decoder = cross_attention_aggregation(i=i , ca_init_map_list = ca_init_map_list, attention_maps_1 = attention_maps_decoder_1 , attention_maps_2=attention_maps_decoder_2, class_one=class_one , from_where= ["down"] , num_pixels=num_pixels , res=res , is_cross = is_cross , select = select , target_location = target_location)

        # ca_init_map_init_step , out_init_step , out_cross_norm_init_step = cross_attention_aggregation(i=i , ca_init_map_list = ca_init_map_list, attention_maps_1 = attention_maps_1 , attention_maps_2=attention_maps_2, class_one=class_one , from_where=from_where , num_pixels=num_pixels , res=res , is_cross = is_cross , select = select , target_location = target_location)
        # ca_init_map_last_step , out_last_step , out_cross_norm_last_step = cross_attention_aggregation(i=i , ca_init_map_list = ca_init_map_list, attention_maps_1 = attention_maps_1 , attention_maps_2=attention_maps_2, class_one=class_one , from_where=from_where , num_pixels=num_pixels , res=res , is_cross = is_cross , select = select , target_location = target_location)

        # out = []
        # out_cross_norm = []
        # if i == 0:
        #     if ca_init_map_list is not None:
        #         ca_init_map = ca_init_map_list[i]
        #     attention_maps = attention_maps_1
        # else:                
        #     if ca_init_map_list is not None:
        #         ca_init_map = ca_init_map_list[i]
        #     attention_maps = attention_maps_2
        
        # if class_one == 'pottedplant' or class_one == 'tvmonitor' or class_one == 'cell phone' or class_one == 'parking meter':
        #     for location in from_where:
        #         for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
        #             if item.shape[1] == num_pixels:
        #                 cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[select]
        #                 cross_maps_norm = torch.sum(cross_maps[:,:,:,target_location] , dim=0)
        #                 if cross_maps_norm.var() > 1e-4:
        #                     cross_maps_norm = (cross_maps_norm - cross_maps_norm.min()) / (cross_maps_norm.max() - cross_maps_norm.min())
        #                     out.append(cross_maps)
        #                     out_cross_norm.append(cross_maps_norm.unsqueeze(0))
        #                 else:
        #                     cross_maps_norm = (cross_maps_norm - cross_maps_norm.min()) / (cross_maps_norm.max() - cross_maps_norm.min())
        #                     cross_maps_norm = cross_maps_norm*0.01
        #                     out.append(cross_maps)
        #                     out_cross_norm.append(cross_maps_norm.unsqueeze(0))
        #     target_location = target_location + 2
        #     for location in from_where:
        #         for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
        #             if item.shape[1] == num_pixels:
        #                 cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[select]
        #                 cross_maps_norm = torch.sum(cross_maps[:,:,:,target_location] , dim=0)
        #                 if cross_maps_norm.var() > 1e-4:
        #                     cross_maps_norm = (cross_maps_norm - cross_maps_norm.min()) / (cross_maps_norm.max() - cross_maps_norm.min())
        #                     out.append(cross_maps)
        #                     out_cross_norm.append(cross_maps_norm.unsqueeze(0))
        #                 else:
        #                     cross_maps_norm = (cross_maps_norm - cross_maps_norm.min()) / (cross_maps_norm.max() - cross_maps_norm.min())
        #                     cross_maps_norm = cross_maps_norm*0.01
        #                     out.append(cross_maps)
        #                     out_cross_norm.append(cross_maps_norm.unsqueeze(0))
        # elif class_one == 'diningtable':
        #     target_location = target_location + 1
        #     for location in from_where:
        #         for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
        #             if item.shape[1] == num_pixels:
        #                 cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[select]
        #                 cross_maps_norm = torch.sum(cross_maps[:,:,:,target_location] , dim=0)
        #                 if cross_maps_norm.var() > 1e-4:
        #                     cross_maps_norm = (cross_maps_norm - cross_maps_norm.min()) / (cross_maps_norm.max() - cross_maps_norm.min())
        #                     out.append(cross_maps)
        #                     out_cross_norm.append(cross_maps_norm.unsqueeze(0))
        #                 else:
        #                     cross_maps_norm = (cross_maps_norm - cross_maps_norm.min()) / (cross_maps_norm.max() - cross_maps_norm.min())
        #                     cross_maps_norm = cross_maps_norm*0.01
        #                     out.append(cross_maps)
        #                     out_cross_norm.append(cross_maps_norm.unsqueeze(0))
        # else:
        #     for location in from_where:
        #         for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
        #             if item.shape[1] == num_pixels:
        #                 cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[select]
        #                 cross_maps_norm = torch.sum(cross_maps[:,:,:,target_location] , dim=0)
        #                 if cross_maps_norm.var() > 1e-4:
        #                     cross_maps_norm = (cross_maps_norm - cross_maps_norm.min()) / (cross_maps_norm.max() - cross_maps_norm.min())
        #                     out.append(cross_maps)
        #                     out_cross_norm.append(cross_maps_norm.unsqueeze(0))
        #                 else:
        #                     cross_maps_norm = (cross_maps_norm - cross_maps_norm.min()) / (cross_maps_norm.max() - cross_maps_norm.min())
        #                     cross_maps_norm = cross_maps_norm*0.01
        #                     out.append(cross_maps)
        #                     out_cross_norm.append(cross_maps_norm.unsqueeze(0))

        
        out = torch.cat(out, dim=0)
        out_cross_norm = torch.cat(out_cross_norm, dim=0)

        class_save_dir = os.path.join(save_dir , class_one)
        if os.path.exists(class_save_dir):
            if state == 'base':
                torch.save(out_cross_norm , os.path.join(class_save_dir , f'ca_{res}_{image_cnt}.pth'))
                # torch.save(out_cross_norm_encoder , os.path.join(class_save_dir , f'ca_{res}_{image_cnt}_encoder.pth'))
                # torch.save(out_cross_norm_decoder , os.path.join(class_save_dir , f'ca_{res}_{image_cnt}_decoder.pth'))

            elif state == 'init':
                torch.save(out_cross_norm , os.path.join(class_save_dir , f'ca_{res}_{image_cnt}_init_timestep.pth'))
            elif state == 'last':
                torch.save(out_cross_norm , os.path.join(class_save_dir , f'ca_{res}_{image_cnt}_last_timestep.pth'))
        else:
            os.makedirs(os.path.join(class_save_dir , class_one))
            if state == 'base':
                torch.save(out_cross_norm , os.path.join(class_save_dir , f'ca_{res}_{image_cnt}.pth'))
                # torch.save(out_cross_norm_encoder , os.path.join(class_save_dir , f'ca_{res}_{image_cnt}_encoder.pth'))
                # torch.save(out_cross_norm_decoder , os.path.join(class_save_dir , f'ca_{res}_{image_cnt}_decoder.pth'))
            elif state == 'init':
                torch.save(out_cross_norm , os.path.join(class_save_dir , f'ca_{res}_{image_cnt}_init_timestep.pth'))
            elif state == 'last':
                torch.save(out_cross_norm , os.path.join(class_save_dir , f'ca_{res}_{image_cnt}_last_timestep.pth'))
        
        if i == 0:
            out_self = self_attention_aggregation(from_where=from_where , attention_maps=attention_maps , is_cross=is_cross , num_pixels = num_pixels)
            # if res == 8:
            #     pass
            # else:
                # out_self_enc = self_attention_aggregation(from_where=["up"] , attention_maps=attention_maps_encoder_1 , is_cross=is_cross , num_pixels = num_pixels)
                # out_self_dec = self_attention_aggregation(from_where=["down"] , attention_maps=attention_maps_decoder_1 , is_cross=is_cross , num_pixels = num_pixels)
        else:
            out_self = self_attention_aggregation(from_where=from_where , attention_maps=attention_maps_2 , is_cross=is_cross , num_pixels = num_pixels)
            # if res == 8:
            #     pass
            # else:
                # out_self_enc = self_attention_aggregation(from_where=["up"] , attention_maps=attention_maps_encoder_2 , is_cross=is_cross , num_pixels = num_pixels)
                # out_self_dec = self_attention_aggregation(from_where=["down"]  , attention_maps=attention_maps_decoder_2 , is_cross=is_cross , num_pixels = num_pixels)
        # if mask is None:
        #     if res == 64:
        #         ca_map = torch.sum(out_cross_norm , dim=0)
        #         ca_map_norm = (ca_map - ca_map.min()) / (ca_map.max() - ca_map.min())
        #         ca_init_map = torch.tensor(ca_init_map).to(out_cross_norm.device)
        #         ca_int_map = (ca_init_map*ca_init_map).reshape(res,res)
        #         ca_int_map = np.array((ca_int_map*255).to('cpu')).astype(np.uint16)
        #         ca_map_mask = ca_init_map > 0.4
        #         ca_map_mask = torch.tensor(np.array(ca_map_mask.to('cpu')).astype(np.uint8)).reshape(res*res)
        #         ca_map_mask = ca_map_mask > 0
        #         ca_map_mask = 1 - ca_map_mask*1
        #         top_idx = torch.tensor(torch.where(ca_map_mask == True)[0]) 

        #     elif res==32:
        #         ca_map = torch.sum(out_cross_norm , dim=0)
        #         ca_map_norm = (ca_map - ca_map.min()) / (ca_map.max() - ca_map.min())
                
        #         ca_init_map = torch.tensor(ca_init_map).to(out_cross_norm.device)
        #         # Otsu Dynamic Thresholding
        #         ca_int_map = (ca_init_map*ca_init_map).reshape(res,res)
        #         ca_int_map = np.array((ca_int_map*255).to('cpu')).astype(np.uint16)
        #         # ret, ca_map_mask = cv2.threshold(ca_int_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #         ca_map_mask = ca_init_map > 0.4
        #         ca_map_mask = torch.tensor(np.array(ca_map_mask.to('cpu')).astype(np.uint8)).reshape(res*res)
        #         ca_map_mask = ca_map_mask > 0
        #         ca_map_mask = 1 - ca_map_mask*1
        #         top_idx = torch.tensor(torch.where(ca_map_mask == True)[0])
        #         ca_map = torch.sum(out_cross_norm , dim=0)
        #         ca_map_norm = (ca_map - ca_map.min()) / (ca_map.max() - ca_map.min())
        #         ca_map_reshape = ca_map_norm.reshape(num_pixels)
        #         ca_map_mask = ca_map_reshape > 0.4
        #         top_idx_cg = torch.tensor(np.where(ca_map_mask.to('cpu') == True)[0])

        #         ca_init_map = torch.tensor(ca_init_map).to(out_cross_norm.device)
        #         ca_map_reshape = ca_init_map.reshape(num_pixels)
        #         ca_map_mask = ca_map_reshape > 0.4
        #         top_idx_cascade = torch.tensor(torch.where(ca_map_mask == True)[0])

        #     else:
        #         ca_map = torch.sum(out_cross_norm , dim=0)
        #         ca_map_norm = (ca_map - ca_map.min()) / (ca_map.max() - ca_map.min())
        #         ca_map_reshape = ca_map_norm.reshape(num_pixels)
        #         # ca_map_mask = ca_map_reshape > 1.2*ca_map_reshape.mean()
        #         # ca_map_mask = ca_map_reshape > 2*ca_map_reshape.mean()
                
        #         # Origin Code
        #         ca_map_mask = ca_map_reshape*ca_map_reshape > 0.5

        #         # Otsu 를 적용한 dynamic threshold 
        #         ca_int_map = (ca_map_reshape*ca_map_reshape).reshape(res,res)
        #         ca_int_map = np.array((ca_int_map*255).to('cpu')).astype(np.uint16)
        #         ret, ca_map_mask = cv2.threshold(ca_int_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #         ca_map_mask = torch.tensor(ca_map_mask.astype(np.uint8)).reshape(res*res)
        #         ca_map_mask = ca_map_mask > 0
        #         top_idx = torch.tensor(torch.where(ca_map_mask == True)[0])
        #         del ca_map_mask , ca_map

        # else:
        #     ca_map = torch.sum(out_cross_norm , dim=0)
        #     ca_map_norm = (ca_map - ca_map.min()) / (ca_map.max() - ca_map.min())
        #     mask = (mask - mask.min()) / (mask.max() - mask.min())
        #     ca_map_norm = (ca_map_norm + mask.to(ca_map_norm.device))/2.0

        #     ca_map_reshape = ca_map_norm.reshape(num_pixels)
        #     ca_map_mask = ca_map_reshape > 1.5*ca_map_reshape.mean()
        #     eroded_mask = cv2.erode(np.array(ca_map_mask.detach().to('cpu')).astype(np.uint8)*255 , np.ones((2,2), np.uint8))
        #     eroded_mask = np.array(eroded_mask == 255)
        #     top_idx = torch.tensor(np.where(eroded_mask == True)[0])
        #     del ca_map_mask , ca_map
        
        # out_self = []
        # for location in from_where:
        #     for item in attention_maps[f"{location}_{'self' if is_cross else 'cross'}"]:
        #         if item.shape[1] == num_pixels:
        #             self_maps = torch.sum(item , dim=0)
        #             out_self.append(self_maps)

        sa_out = out_self[0]
        if state == 'base':
            # png_save(self_attn_map = sa_out , res=res , image_cnt = image_cnt , png_save_dir = os.path.join(class_save_dir , f'sa_{res}_{image_cnt}.png') )
            # torch.save(sa_out , os.path.join(class_save_dir , f'sa_{res}_{image_cnt}.pth'))
            if res == 64:
                png_save(self_attn_map = sa_out , res=res , image_cnt = image_cnt , png_save_dir = os.path.join(class_save_dir , f'sa_{res}_{image_cnt}.png') )
            else:
                torch.save(sa_out , os.path.join(class_save_dir , f'sa_{res}_{image_cnt}.pth'))

            # if res != 8:
            #     sa_out_enc = out_self_enc[0]
            #     sa_out_dec = out_self_dec[0]
            #     png_save(self_attn_map = sa_out_enc , res=res , image_cnt = image_cnt , png_save_dir = os.path.join(class_save_dir , f'sa_{res}_{image_cnt}_enc.png') )
            #     png_save(self_attn_map = sa_out_dec , res=res , image_cnt = image_cnt , png_save_dir = os.path.join(class_save_dir , f'sa_{res}_{image_cnt}_dec.png') )
                # torch.save(sa_out_enc , os.path.join(class_save_dir , f'sa_{res}_{image_cnt}_enc.pth'))
                # torch.save(sa_out_dec , os.path.join(class_save_dir , f'sa_{res}_{image_cnt}_dec.pth'))
        elif state == 'init':
            png_save(self_attn_map = sa_out , res=res , image_cnt = image_cnt , png_save_dir = os.path.join(class_save_dir , f'sa_{res}_{image_cnt}_init_timestep.png') )
            # torch.save(sa_out , os.path.join(class_save_dir , f'sa_{res}_{image_cnt}_init_timestep.pth'))
            del sa_out, out_self
        elif state == 'last':
            png_save(self_attn_map = sa_out , res=res , image_cnt = image_cnt , png_save_dir = os.path.join(class_save_dir , f'sa_{res}_{image_cnt}_last_timestep.png') )
            # torch.save(sa_out , os.path.join(class_save_dir , f'sa_{res}_{image_cnt}_last_timestep.pth'))
            del sa_out, out_self
        
        # if state == 'base':
        #     top_idx = seed_extraction(mask=mask , res = res , out_cross_norm=out_cross_norm , ca_init_map = ca_init_map , num_pixels = num_pixels)        
        #     cos = torch.nn.CosineSimilarity(dim=1)
        #     vis_map = torch.zeros(out_self[0].shape[0])
        #     vis_map = region_expansion(top_idx=top_idx , out_self=out_self , cos=cos , res = res , ca_init_map=ca_init_map , vis_map=vis_map)
        #     out_list.append(out.cpu())
        #     vis_map_list.append(vis_map.cpu())
        #     del out, vis_map, out_self

        # for idx in top_idx:
        #     anchor_token = out_self[0][idx]
        #     sim_embedding = cos(out_self[0] , anchor_token.unsqueeze(0)) 
        #     sim_embedding = sim_embedding
        #     vis_map = vis_map + sim_embedding.detach().to('cpu')
        
        # if res > 31:
        #     vis_map = 1 - vis_map
        #     vis_map = vis_map.reshape(res , res)
        #     vis_map = (vis_map - vis_map.min()) / (vis_map.max() - vis_map.min())
        #     ca_init_map_norm = (ca_init_map - ca_init_map.min()) / (ca_init_map.max() - ca_init_map.min())
        #     vis_map = ca_init_map_norm.to('cpu') * vis_map
        #     vis_map = (vis_map - vis_map.min()) / (vis_map.max() - vis_map.min())
        # elif res == 16:
        #     vis_map = ( vis_map - vis_map.min()) / (vis_map.max() - vis_map.min())
        #     vis_map = vis_map.reshape(res , res)
        # else:
        #     vis_map = vis_map.reshape(res , res)
            

    del attention_maps
    torch.cuda.empty_cache()
    return out_list , vis_map_list


def mask_image(image, mask_2d, rgb=None, valid = False):
    h, w = mask_2d.shape

    mask_3d_color = np.zeros((h, w, 3), dtype="uint8")
    
        
    image.astype("uint8")
    mask = (mask_2d!=0).astype(bool)
    if rgb is None:
        rgb = np.random.randint(0, 255, (1, 3), dtype=np.uint8)
        
    mask_3d_color[mask_2d[:, :] == 1] = rgb
    image[mask] = image[mask] * 0.2 + mask_3d_color[mask] * 0.8
    
    if valid:
        mask_3d_color[mask_2d[:, :] == 1] = [[0,0,0]]
        kernel = np.ones((5,5),np.uint8)  
        mask_2d = cv2.dilate(mask_2d,kernel,iterations = 4)
        mask = (mask_2d!=0).astype(bool)
        image[mask] = image[mask] * 0 + mask_3d_color[mask] * 1
        return image,rgb
        
    return image,rgb

def get_findContours(mask):
    mask_instance = (mask>0.5 * 1).astype(np.uint8) 
    ontours, hierarchy = cv2.findContours(mask_instance.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    min_area = 0
    polygon_ins = []
    x,y,w,h = 0,0,0,0
    
    image_h, image_w = mask.shape[0:2]
    gt_kernel = np.zeros((image_h,image_w), dtype='uint8')
    for cnt in ontours:
        x_ins_t, y_ins_t, w_ins_t, h_ins_t = cv2.boundingRect(cnt)

        if w_ins_t*h_ins_t<250:
            continue
        cv2.fillPoly(gt_kernel, [cnt], 1)

    return gt_kernel

def save_cross_attention_batch(original_image,attention_store: AttentionStore, init_attention_store: AttentionStore, last_attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0,out_put="./test_1.jpg",image_cnt=0,class_one=None,prompts=None , tokenizer=None,mask_diff=None , save_dir =None , target_class = None , attn_save_dir = None):
    device = attention_store.get_average_attention()['down_cross'][0].device
    original_image = original_image.copy()
    show = True
    target_token = tokenizer.encode(target_class)
    tokens = tokenizer.encode(prompts[select])
    if len(target_token) == 3:
        target_location = np.where(np.array(tokens) == target_token[1])[0][0]
    else:
        target_location = np.where(np.array(tokens) == target_token[1])[0][0] + 1
    decoder = tokenizer.decode
    class_one = target_class

    attention_maps_8s , sa_agg_map_8 = aggregate_attention_batch(attention_store, 8, ("up", "mid", "down"), True, select,prompts=prompts , mask=None , ca_init_map_list=None , class_one = class_one , target_location = target_location , image_cnt = image_cnt , save_dir = attn_save_dir , state = 'base')
    
    attention_maps , sa_agg_map_16 = aggregate_attention_batch(attention_store, 16, from_where, True, select,prompts=prompts, mask=None , ca_init_map_list=None , class_one = class_one , target_location = target_location , image_cnt = image_cnt , save_dir = attn_save_dir, state = 'base')
    del attention_maps
    
    attention_maps_32 , sa_agg_map_32 = aggregate_attention_batch(attention_store, 32, from_where, True, select,prompts=prompts, mask=None, ca_init_map_list=None, class_one = class_one , target_location = target_location , image_cnt = image_cnt , save_dir = attn_save_dir, state = 'base')
    del attention_maps_32

    attention_maps_64 , sa_agg_map_64 = aggregate_attention_batch(attention_store, 64, from_where, True, select,prompts=prompts , mask = None, ca_init_map_list=None, class_one = class_one , target_location = target_location , image_cnt = image_cnt , save_dir = attn_save_dir, state = 'base')
    del sa_agg_map_64 , sa_agg_map_32 , sa_agg_map_16 , attention_store , attention_maps_64
    torch.cuda.empty_cache()
    

def save_cross_attention(original_image,attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0,out_put=None,image_cnt=0,class_one=None,prompts=None , tokenizer=None,mask_diff=None , save_dir =None , target_class = None , attn_save_dir = None):
    
#     ("up", "down")
#     ("mid", "up", "down")
    device = attention_store.get_average_attention()['down_cross'][0].device
    original_image = original_image.copy()
    target_token = tokenizer.encode(target_class)
    tokens = tokenizer.encode(prompts[select])
    target_location = np.where(np.array(tokens) == target_token[1])[0][0]
    
    class_one = target_class
    
    
    # "up", "down"
    attention_maps_8s , sa_agg_map_8 = aggregate_attention(attention_store, 8, ("up", "mid", "down"), True, select,prompts=prompts , mask=None , ca_init_map=None , class_one = class_one , target_location = target_location , image_cnt = image_cnt , save_dir = attn_save_dir)
    attention_maps_8s = attention_maps_8s.sum(0) / attention_maps_8s.shape[0]
    
    
    attention_maps , sa_agg_map_16 = aggregate_attention(attention_store, 16, from_where, True, select,prompts=prompts, mask=None , ca_init_map=None , class_one = class_one , target_location = target_location , image_cnt = image_cnt , save_dir = attn_save_dir)
    attention_maps = attention_maps.sum(0) / attention_maps.shape[0]

    attn_mask_for_64 = torch.tensor(cv2.resize(sa_agg_map_16.numpy(), (64, 64), interpolation=cv2.INTER_CUBIC))
    # attn_mask_for_64 = np.array(attn_mask_for_64 > 1.5*attn_mask_for_64.mean())

    # eroded_mask = cv2.erode( (attn_mask_for_64).astype(np.uint8)*255 , np.ones((3,3), np.uint8))
    # eroded_mask = np.array(eroded_mask == 255)
    ca_map_for_32 = cv2.resize(sa_agg_map_16.numpy(), (32, 32), interpolation=cv2.INTER_CUBIC)
    attention_maps_32 , sa_agg_map_32 = aggregate_attention(attention_store, 32, from_where, True, select,prompts=prompts, mask=None, ca_init_map=ca_map_for_32, class_one = class_one , target_location = target_location , image_cnt = image_cnt , save_dir = attn_save_dir)
    attention_maps_32 = attention_maps_32.sum(0) / attention_maps_32.shape[0]

    # ca_map_for_64_1 = cv2.resize(sa_agg_map_16.numpy(), (64, 64), interpolation=cv2.INTER_CUBIC)
    ca_map_for_64 = cv2.resize(sa_agg_map_32.numpy(), (64, 64), interpolation=cv2.INTER_CUBIC)
    # ca_map_for_64 = (ca_map_for_64_1 + ca_map_for_64_2)/2
    attention_maps_64 , sa_agg_map_64 = aggregate_attention(attention_store, 64, ["up"], True, select,prompts=prompts , mask = None, ca_init_map=ca_map_for_64, class_one = class_one , target_location = target_location , image_cnt = image_cnt , save_dir = attn_save_dir)
    attention_maps_64 = attention_maps_64.sum(0) / attention_maps_64.shape[0]


    
    sa_agg_map_8_reshape = cv2.resize(sa_agg_map_8.numpy(), (512, 512), interpolation=cv2.INTER_CUBIC)
    sa_agg_map_16_reshape = cv2.resize(sa_agg_map_16.numpy(), (512, 512), interpolation=cv2.INTER_CUBIC)
    sa_agg_map_32_reshape = cv2.resize(sa_agg_map_32.numpy(), (512, 512), interpolation=cv2.INTER_CUBIC)
    sa_agg_map_64_reshape = cv2.resize(sa_agg_map_64.numpy(), (512, 512), interpolation=cv2.INTER_CUBIC)


    sa_agg_map_8_reshape = (sa_agg_map_8_reshape-sa_agg_map_8_reshape.min()) / (sa_agg_map_8_reshape.max()-sa_agg_map_8_reshape.min())
    sa_agg_map_16_reshape = (sa_agg_map_16_reshape-sa_agg_map_16_reshape.min()) / (sa_agg_map_16_reshape.max() - sa_agg_map_16_reshape.min())
    sa_agg_map_32_reshape = (sa_agg_map_32_reshape-sa_agg_map_32_reshape.min()) / (sa_agg_map_32_reshape.max() - sa_agg_map_32_reshape.min())
    
    sa_agg_map_64_reshape = (sa_agg_map_64_reshape - sa_agg_map_64_reshape.min()) / (sa_agg_map_64_reshape.max() - sa_agg_map_64_reshape.min())
    sa_agg_map_16_mask = sa_agg_map_16_reshape > 0.3
    sa_agg_map_total = ( sa_agg_map_32_reshape + sa_agg_map_64_reshape) / 2.0
    sa_agg_map_mask = sa_agg_map_64_reshape > 0.3
    

    # PAMR
    input_image_tensor = torch.tensor(np.array(original_image)).permute(2,0,1).unsqueeze(0).type(torch.float32)
    mask_tensor = torch.tensor(sa_agg_map_mask).unsqueeze(0).unsqueeze(0)
    pamr_operation = PAMR(num_iter=5000)
    pamr_tensor = pamr_operation(input_image_tensor  , mask_tensor*1)[0][0]
    mask_output = (pamr_tensor - pamr_tensor.min()) / (pamr_tensor.max() - pamr_tensor.min())
    mask_output = (mask_output > 0.8)*255
    
    # Device Change , PAMR to GPU
    
    input_image_tensor = torch.cat((input_image_tensor.to(device) , input_image_tensor.to(device) , input_image_tensor.to(device) ) , dim=0)
    
    # Make PIL
    original_image = Image.fromarray(original_image)
    mask_output = Image.fromarray(mask_output.cpu().numpy().astype(np.uint8) , mode='L')

    return original_image , mask_output




    
    

def show_self_attention_comp(attention_store: AttentionStore, res: int, from_where: List[str],
                        max_com=10, select: int = 0):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils.view_images(np.concatenate(images, axis=1))
    
    
def run(prompts, controller, latent=None, generator=None,out_put = None,ldm_stable=None):
    images_here, x_t = ptp_utils.text2image_ldm_stable_seediff(ldm_stable, prompts, controller, latent=latent, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=7, generator=generator, low_resource=LOW_RESOURCE)
    # ptp_utils.view_images(images_here,out_put = out_put)
    return images_here, x_t

def sub_processor(class_name):
    LOW_RESOURCE = False 
    NUM_DIFFUSION_STEPS = 50
    GUIDANCE_SCALE = 7.5
    MAX_NUM_WORDS = 77
    device = f'cuda:2' if torch.cuda.is_available() else torch.device('cpu')
    ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token='SeeDiff').to(device)    
    tokenizer = ldm_stable.tokenizer
    
    image_cnt = 200000    
    
    init_prompt = f"A photo of a {class_name}"
    prompts = [init_prompt]

    controller = AttentionStore()
    g_cpu = torch.Generator().manual_seed(image_cnt)
    seed_list = [g_cpu]

    image, x_t= run(prompts, controller, latent=None,  generator=seed_list[0], out_put = None,ldm_stable=ldm_stable)

    origin_img , mask = save_cross_attention(image[0].copy(),controller, res=32, from_where=("up", "down"),out_put = None,image_cnt=image_cnt,class_one=class_name,prompts=prompts,tokenizer=tokenizer ,save_dir = None , target_class = class_name , attn_save_dir = None)
    del controller
    return origin_img, mask
    
    
    # if len(prompts) > 1:
    #     save_cross_attention_batch(image[0].copy(),controller, res=32, from_where=("up", "down"),out_put = os.path.join(npy_path,"image_{}_{}".format(args.classes,image_cnt)),image_cnt=image_cnt,class_one=args.classes,prompts=prompts,tokenizer=tokenizer ,save_dir = args.save_dir , target_class = args.classes , attn_save_dir = attn_save_dir , init_attention_store=None , last_attention_store=None)
    #     image_cnt = image_cnt + len(prompts)
    #     del controller
    #     torch.cuda.empty_cache()
    # else:        
    #     save_cross_attention_batch(image[0].copy(),controller, res=32, from_where=("up", "down"),out_put = os.path.join(npy_path,"image_{}_{}".format(args.classes,image_cnt)),image_cnt=image_cnt,class_one=args.classes,prompts=prompts,tokenizer=tokenizer ,save_dir = args.save_dir , target_class = args.classes , attn_save_dir = attn_save_dir , init_attention_store=None , last_attention_store=None)
    #     image_cnt = image_cnt + 1 




    
if __name__ == '__main__':
    
    # origin_img , mask = sub_processor('dog')
    app = gr.Interface(
    fn=sub_processor,
    inputs=gr.Textbox(label="Enter a class. (e.g. dog, cat, car)"),
    outputs=[gr.Image(label="Synthetic Image", type="pil"),
    gr.Image(label="Synthetic Mask", type="pil")
    ],
    title="SeeDiff : Demo page",
    description="Input class as a text then SeeDiff will generate the image and mask."
)
    app.launch(
        server_name="0.0.0.0",
        server_port=8020,
        share=True    
    )   




