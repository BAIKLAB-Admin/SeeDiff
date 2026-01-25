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

    class_save_dir = os.path.join(save_dir , class_one)
    if os.path.exists(class_save_dir):
        torch.save(out_cross_norm , os.path.join(class_save_dir , f'ca_{res}_{image_cnt}.pth'))
    else:
        os.makedirs(os.path.join(class_save_dir , class_one))
        torch.save(out_cross_norm , os.path.join(class_save_dir , f'ca_{res}_{image_cnt}.pth'))

    # Cross-attention coordinates
    if mask is None:
        if res == 64:
            # res_64 origin code
            ca_map = torch.sum(out_cross_norm , dim=0)
            ca_map_norm = (ca_map - ca_map.min()) / (ca_map.max() - ca_map.min())



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


        elif res==32:
            ca_map = torch.sum(out_cross_norm , dim=0)
            ca_map_norm = (ca_map - ca_map.min()) / (ca_map.max() - ca_map.min())            
            ca_init_map = torch.tensor(ca_init_map).to(out_cross_norm.device)
            
            # Otsu Dynamic Thresholding
            ca_int_map = (ca_init_map*ca_init_map).reshape(res,res)
            ca_int_map = np.array((ca_int_map*255).to('cpu')).astype(np.uint16)
            ret, ca_map_mask = cv2.threshold(ca_int_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ca_map_mask = torch.tensor(ca_map_mask.astype(np.uint8)).reshape(res*res)
            ca_map_mask = ca_map_mask > 0
            ca_map_mask = 1 - ca_map_mask*1
            top_idx = torch.tensor(torch.where(ca_map_mask == True)[0])

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

            ca_int_map = (ca_map_reshape*ca_map_reshape).reshape(res,res)
            ca_int_map = np.array((ca_int_map*255).to('cpu')).astype(np.uint16)
            _, ca_map_mask = cv2.threshold(ca_int_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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



    # Self-attention aggregation
    out_self = []
    for location in from_where:
        for item in attention_maps[f"{location}_{'self' if is_cross else 'cross'}"]:
            if item.shape[1] == num_pixels:
                self_maps = torch.sum(item , dim=0)
                out_self.append(self_maps)

    sa_out = out_self[0]
    torch.save(sa_out , os.path.join(class_save_dir , f'sa_{res}_{image_cnt}.pth'))

    cos = torch.nn.CosineSimilarity(dim=1)
    vis_map = torch.zeros(out_self[0].shape[0])
    for idx in top_idx:
        anchor_token = out_self[0][idx]
        sim_embedding = cos(out_self[0] , anchor_token.unsqueeze(0)) 
        sim_embedding = sim_embedding
        vis_map = vis_map + sim_embedding.detach().to('cpu')
    
    if res > 31:
        if res == 64 or res == 32:  
            vis_map = 1 - vis_map
            vis_map = vis_map.reshape(res , res)
            vis_map = (vis_map - vis_map.min()) / (vis_map.max() - vis_map.min())
            ca_init_map_norm = (ca_init_map - ca_init_map.min()) / (ca_init_map.max() - ca_init_map.min())
            vis_map = ca_init_map_norm.to('cpu') * vis_map
            vis_map = (vis_map - vis_map.min()) / (vis_map.max() - vis_map.min())
        if res ==32:
            return out.cpu() , vis_map.cpu() 
        elif res == 64:
            return out.cpu() , vis_map.cpu() 




    elif res == 16:
        vis_map = ( vis_map - vis_map.min()) / (vis_map.max() - vis_map.min())
        vis_map = vis_map.reshape(res , res)
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
            ca_int_map = (ca_init_map*ca_init_map).reshape(res,res)
            ca_int_map = np.array((ca_int_map*255).to('cpu')).astype(np.uint16)
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
            ca_map_mask = ca_map_reshape*ca_map_reshape > 0.5
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
    num_pixels = res ** 2
    target_location = target_location
    batch_size = len(prompts)
    out_list = []
    vis_map_list = []
    for i in range(batch_size):
        image_cnt = image_cnt + i
        ca_init_map , out , out_cross_norm = cross_attention_aggregation(i=i , ca_init_map_list = ca_init_map_list, attention_maps = attention_maps, class_one=class_one , from_where=from_where , num_pixels=num_pixels , res=res , is_cross = is_cross , select = select , target_location = target_location)    
        

        
        out = torch.cat(out, dim=0)
        out_cross_norm = torch.cat(out_cross_norm, dim=0)

        class_save_dir = os.path.join(save_dir , class_one)
        if os.path.exists(class_save_dir):
            if state == 'base':
                torch.save(out_cross_norm , os.path.join(class_save_dir , f'ca_{res}_{image_cnt}.pth'))

            elif state == 'init':
                torch.save(out_cross_norm , os.path.join(class_save_dir , f'ca_{res}_{image_cnt}_init_timestep.pth'))
            elif state == 'last':
                torch.save(out_cross_norm , os.path.join(class_save_dir , f'ca_{res}_{image_cnt}_last_timestep.pth'))
        else:
            os.makedirs(os.path.join(class_save_dir , class_one))
            if state == 'base':
                torch.save(out_cross_norm , os.path.join(class_save_dir , f'ca_{res}_{image_cnt}.pth'))
            elif state == 'init':
                torch.save(out_cross_norm , os.path.join(class_save_dir , f'ca_{res}_{image_cnt}_init_timestep.pth'))
            elif state == 'last':
                torch.save(out_cross_norm , os.path.join(class_save_dir , f'ca_{res}_{image_cnt}_last_timestep.pth'))
        
        if i == 0:
            out_self = self_attention_aggregation(from_where=from_where , attention_maps=attention_maps , is_cross=is_cross , num_pixels = num_pixels)

        else:
            out_self = self_attention_aggregation(from_where=from_where , attention_maps=attention_maps_2 , is_cross=is_cross , num_pixels = num_pixels)

        sa_out = out_self[0]
        if state == 'base':
            if res == 64:
                png_save(self_attn_map = sa_out , res=res , image_cnt = image_cnt , png_save_dir = os.path.join(class_save_dir , f'sa_{res}_{image_cnt}.png') )
            else:
                torch.save(sa_out , os.path.join(class_save_dir , f'sa_{res}_{image_cnt}.pth'))

        elif state == 'init':
            png_save(self_attn_map = sa_out , res=res , image_cnt = image_cnt , png_save_dir = os.path.join(class_save_dir , f'sa_{res}_{image_cnt}_init_timestep.png') )
            del sa_out, out_self
        elif state == 'last':
            png_save(self_attn_map = sa_out , res=res , image_cnt = image_cnt , png_save_dir = os.path.join(class_save_dir , f'sa_{res}_{image_cnt}_last_timestep.png') )
            del sa_out, out_self
                          
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
    

def save_cross_attention(original_image,attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0,out_put="./test_1.jpg",image_cnt=0,class_one=None,prompts=None , tokenizer=None,mask_diff=None , save_dir =None , target_class = None , attn_save_dir = None):
    
#     ("up", "down")
#     ("mid", "up", "down")
    device = attention_store.get_average_attention()['down_cross'][0].device
    original_image = original_image.copy()
    show = True
    target_token = tokenizer.encode(target_class)
    tokens = tokenizer.encode(prompts[select])
    target_location = np.where(np.array(tokens) == target_token[1])[0][0]
    decoder = tokenizer.decode
    class_one = target_class

    attention_maps_8s , sa_agg_map_8 = aggregate_attention(attention_store, 8, ("up", "mid", "down"), True, select,prompts=prompts , mask=None , ca_init_map=None , class_one = class_one , target_location = target_location , image_cnt = image_cnt , save_dir = attn_save_dir)
    attention_maps_8s = attention_maps_8s.sum(0) / attention_maps_8s.shape[0]
    
    
    attention_maps , sa_agg_map_16 = aggregate_attention(attention_store, 16, from_where, True, select,prompts=prompts, mask=None , ca_init_map=None , class_one = class_one , target_location = target_location , image_cnt = image_cnt , save_dir = attn_save_dir)
    attention_maps = attention_maps.sum(0) / attention_maps.shape[0]

    attn_mask_for_64 = torch.tensor(cv2.resize(sa_agg_map_16.numpy(), (64, 64), interpolation=cv2.INTER_CUBIC))

    ca_map_for_32 = cv2.resize(sa_agg_map_16.numpy(), (32, 32), interpolation=cv2.INTER_CUBIC)
    attention_maps_32 , sa_agg_map_32 = aggregate_attention(attention_store, 32, from_where, True, select,prompts=prompts, mask=None, ca_init_map=ca_map_for_32, class_one = class_one , target_location = target_location , image_cnt = image_cnt , save_dir = attn_save_dir)
    attention_maps_32 = attention_maps_32.sum(0) / attention_maps_32.shape[0]

    ca_map_for_64 = cv2.resize(sa_agg_map_32.numpy(), (64, 64), interpolation=cv2.INTER_CUBIC)
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
    sa_agg_map_mask = sa_agg_map_64_reshape > 0.5
    

    cam_dict = {}
    for idx, class_one in enumerate(coco_category_list):
        
        gt_kernel_final = np.zeros((512,512), dtype='float32')
        number_gt = 0
        for i in range(len(tokens)):
            class_current = decoder(int(tokens[i])) 
            
            category_list_check = VOC_category_list_check[class_one]
 
            image_8 = attention_maps_8s[:, :, i]
            image_8 = cv2.resize(image_8.numpy(), (512, 512), interpolation=cv2.INTER_CUBIC)
            image_8 = image_8 / image_8.max()
            
            image_16 = attention_maps[:, :, i]
            image_16 = cv2.resize(image_16.numpy(), (512, 512), interpolation=cv2.INTER_CUBIC)
            image_16 = image_16 / image_16.max()
            
            image_32 = attention_maps_32[:, :, i]
            image_32 = cv2.resize(image_32.numpy(), (512, 512), interpolation=cv2.INTER_CUBIC)
            image_32 = image_32 / image_32.max()
            
            image_64 = attention_maps_64[:, :, i]
            image_64 = cv2.resize(image_64.numpy(), (512, 512), interpolation=cv2.INTER_CUBIC)
            image_64 = image_64 / image_64.max()
            
            if class_one == "sofa" or class_one == "train" or class_one == "tvmonitor":
                image = image_8
            elif class_one == "diningtable":
                image = image_16
            else:
                image = (image_16 + image_32 + image_64) / 3


            gt_kernel_final += image.copy()
            number_gt += 1

       
    diff_mask_map = (image_16 + image_32 + image_64)/3.0
    diff_mask = (diff_mask_map > 0.5)

    img_name = out_put[out_put.rfind('/')+1:]
    
    # Original Mask save
    origin_mask_sup_dir = os.path.join(save_dir , 'diff_mask')
    if not os.path.isdir(origin_mask_sup_dir):
        os.mkdir(origin_mask_sup_dir) 
    origin_mask_class_dir = os.path.join(origin_mask_sup_dir ,target_class)
    if not os.path.isdir(origin_mask_class_dir):
        os.mkdir(origin_mask_class_dir) 
    origin_mask_dir = os.path.join(origin_mask_class_dir , f'{img_name}.png')  
    cv2.imwrite(origin_mask_dir , diff_mask*255)

    # PAMR
    input_image_tensor = torch.tensor(np.array(original_image)).permute(2,0,1).unsqueeze(0).type(torch.float32)
    mask_tensor = torch.tensor(sa_agg_map_mask).unsqueeze(0).unsqueeze(0)
    diff_mask_tensor = torch.tensor(diff_mask).unsqueeze(0).unsqueeze(0).to(device)
    pamr_operation = PAMR(num_iter=30000)
    a = pamr_operation(input_image_tensor  , mask_tensor*1)
    

    diff_mask_map = (image_16 + image_32 + image_64)/3.0
    neurips_mask_map = (image_16 * 1)
    
    input_image_tensor = torch.cat((input_image_tensor.to(device) , input_image_tensor.to(device) , input_image_tensor.to(device) ) , dim=0)
    
    # Img Save
    gen_img_sup_dir = os.path.join(save_dir , 'generated_dataset')
    if not os.path.isdir(gen_img_sup_dir):
        os.mkdir(gen_img_sup_dir) 
    gen_img_class_dir = os.path.join(gen_img_sup_dir ,target_class)
    if not os.path.isdir(gen_img_class_dir): 
        os.mkdir(gen_img_class_dir) 
    gen_img_output_dir = os.path.join(gen_img_class_dir , 'image')
    if not os.path.isdir(gen_img_output_dir): 
        os.mkdir(gen_img_output_dir) 
    gen_mask_output_dir = os.path.join(gen_img_class_dir , 'mask')
    if not os.path.isdir(gen_mask_output_dir): 
        os.mkdir(gen_mask_output_dir) 

    gen_img_dir = os.path.join(gen_img_output_dir , f'{img_name}.jpg') 
    gen_mask_dir = os.path.join(gen_mask_output_dir , f'{img_name}.png')
    cv2.imwrite(gen_img_dir , cv2.cvtColor(original_image , cv2.COLOR_RGB2BGR))
    cv2.imwrite(gen_mask_dir , cv2.cvtColor(original_image , cv2.COLOR_RGB2BGR))




    
    

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
    
    
def run(prompts, controller, latent=None, generator=None,out_put = "",ldm_stable=None):
    images_here, x_t = ptp_utils.text2image_ldm_stable_seediff(ldm_stable, prompts, controller, latent=latent, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=7, generator=generator, low_resource=LOW_RESOURCE)
    ptp_utils.view_images(images_here,out_put = out_put)
    return images_here, x_t


def clipretrieval(text,check):
    sensitive_word = ["vector","stock","3d","-3d","-","blur","Vector","blurred","shot","close-up","Headlight","Stock","headlights","Defocused","Close-up","3D","cartoon","interior","internal"] 
    prompts = []
    for prompt in tqdm(text):
        try:
#             ClipClient(url="https://knn.laion.ai/knn-service", indice_name="laion5B-L-14")
            client = ClipClient(
                url="https://knn.laion.ai/knn-service",
                indice_name="laion5B-L-14",
                aesthetic_score=9,
                aesthetic_weight=0.5,
                modality=Modality.IMAGE,
                num_images=3000,
            )
            results = client.query(text=prompt)
        except:
            client = ClipClient(
                url="https://knn.laion.ai/knn-service",
                indice_name="laion5B-L-14",
                aesthetic_score=9,
                aesthetic_weight=0.5,
                modality=Modality.IMAGE,
                num_images=1000,
            )
            results = client.query(text=prompt)
            
        for i,line in enumerate(results):
            caption = line["caption"]
            caption_split = caption.split(" ")
            continue_flag = True
            
            for chec in check:
                sen_flag = True
                for c in sensitive_word:
                    if c in caption_split:
                        sen_flag = False

                if chec in caption_split[:5] and sen_flag:
                    continue_flag=False
                    
            if continue_flag:
                continue
                
            if len(caption_split)>50:
                continue

            prompts.append("Photo of "+caption)
    
    return prompts

def sub_processor(pid , args):
    torch.cuda.set_device(pid)
    text = 'processor %d' % pid
    print(text)

    LOW_RESOURCE = False 
    NUM_DIFFUSION_STEPS = 50
    GUIDANCE_SCALE = 7.5
    MAX_NUM_WORDS = 77
    device = f'cuda:{args.gpu_num}' if torch.cuda.is_available() else torch.device('cpu')
    ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=args.MY_TOKEN).to(device)    
    tokenizer = ldm_stable.tokenizer
    number_per_thread_num = int(int(args.image_number)/int(args.thread_num))
    image_cnt = pid * (number_per_thread_num*2) + 200000    
    image_path = os.path.join(args.output,"train_image")
    npy_path = os.path.join(args.output,"npy")
    
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    if not os.path.exists(npy_path):
        os.makedirs(npy_path)
        
    
    for k in range(args.image_number):
        init_prompt = f"A long shot of a {args.classes}"
        prompts = [init_prompt]
        attn_save_dir = args.save_dir
        controller = AttentionStore()
        g_cpu = torch.Generator().manual_seed(image_cnt)
        seed_list = [g_cpu]
        image, x_t= run(prompts, controller, latent=None,  generator=seed_list[0], out_put = os.path.join(image_path,"image_{}_{}.jpg".format(args.classes,image_cnt)),ldm_stable=ldm_stable)
    
        cv2.imwrite('gen_test.png' , cv2.cvtColor(image[0] , cv2.COLOR_RGB2BGR))

        # Create directory
        if not os.path.exists(os.path.join(args.save_dir , 'gen_img')):
            os.mkdir(os.path.join(args.save_dir , 'gen_img'))
        if not os.path.exists(os.path.join(args.save_dir , 'gen_img' , args.classes)):
            os.mkdir(os.path.join(args.save_dir , 'gen_img' ,  args.classes))
        
        for i in range(len(prompts)):
            img_name = image_cnt + i
            cv2.imwrite(os.path.join(args.save_dir , 'gen_img' , args.classes , f'{img_name}.jpg') , cv2.cvtColor(image[i] , cv2.COLOR_RGB2BGR))

        save_cross_attention(image[0].copy(),controller, res=32, from_where=("up", "down"),out_put = os.path.join(npy_path,"image_{}_{}".format(args.classes,image_cnt)),image_cnt=image_cnt,class_one=args.classes,prompts=prompts,tokenizer=tokenizer ,save_dir = args.save_dir , target_class = args.classes , attn_save_dir = attn_save_dir)
        image_cnt = image_cnt + 1 





    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", default="dog", type=str)
    parser.add_argument("--thread_num", default=8, type=int)
    parser.add_argument("--output", default=None, type=str)
    parser.add_argument("--image_number", default=None, type=str)
    parser.add_argument("--MY_TOKEN", default=None, type=str)
    parser.add_argument("--gpu_num", default=None , type=int)
    parser.add_argument("--cc_data_path", default=None , type=str)
    parser.add_argument("--cc_data_base_path", default=None , type=str)
    args = parser.parse_args()    
    args.save_dir = args.output


    result_dict = mp.Manager().dict()
    mp = mp.get_context("spawn")
    processes = []

    print('Start Generation')
    for i in range(args.thread_num):
        p = mp.Process(target=sub_processor, args=(i, args))
        p.start()
        processes.append(p)


    for p in processes:
        p.join()

    result_dict = dict(result_dict)
    




