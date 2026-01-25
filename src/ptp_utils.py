# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Optional, Union, Tuple, List, Callable, Dict
from IPython.display import display
from tqdm.notebook import tqdm
import copy

import torch.nn as nn

def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)
    img[:h,:,:] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img


def view_images(images, num_rows=1, offset_ratio=0.02,out_put="./test_1.jpg"):

    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]
    
#     cv2.imwrite(out_put,image_)
#     print(image_.shape)
    pil_img = Image.fromarray(image_)
    pil_img.save(out_put)
#     display(pil_img)




def diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False, posterior = None):
    
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        # guidance_scale = 100
        # Origin
        if posterior == None:
            latents_noisy = latents
        else:
            # latents_noisy = model.scheduler.add_noise(latents , posterior , t)
        
            latents_noisy = model.scheduler.add_noise(posterior , latents , t)
        latents_input = torch.cat([latents_noisy] * 2)
        # test
        
        # if len(controller.attention_store) > 0 and t > 950:
        #     last_attn = np.array(controller.attention_store['down_cross'][0][:,:,4].to('cpu'))
        # 
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        # 
        # if t < 50:
        #     cur_attn = controller.attention_store['down_cross'][0][:,:,4]
        # 
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        del noise_pred
    
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    
    
    # latents = model.scheduler.step(noise_pred, t, latents_noisy)["prev_sample"]
    
    if posterior == None:
        latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    else:
        # latents = model.scheduler.step(noise_pred, t, latents_noisy)["prev_sample"]
        latents = model.scheduler.step(noise_pred, t, posterior)["prev_sample"]
    
    latents = controller.step_callback(latents)
    del latents_input , noise_pred_uncond , noise_prediction_text
    torch.cuda.empty_cache()
    return latents 

def diffusion_step_seediff(model, controller, latents, context, t, guidance_scale, low_resource=False):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents



def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def init_latent(latent, model, height, width, generator, batch_size):
    latent_list = []
    if batch_size > 1:
        for i in range(batch_size):
            latent = torch.randn((1, model.unet.in_channels, height // 8, width // 8),generator=generator[i],)
            latent_list.append(latent)
        latents = torch.cat( (latent_list[0] , latent_list[1]) , dim=0 ).to(model.device)
    else:
        if latent is None:
            latent = torch.randn(
                (1, model.unet.in_channels, height // 8, width // 8),
                generator=generator,
            )
        latents = latent.expand(batch_size,  model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents


@torch.no_grad()
def text2image_ldm(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
):
    register_attention_control(model, controller)
    height = width = 256
    batch_size = len(prompt)
    
    negative_prompt = (
        "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face,"
        + "out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy,"
        + "watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face"
    )

    uncond_input = model.tokenizer([negative_prompt] * batch_size, padding="max_length", max_length=77, return_tensors="pt")
    uncond_embeddings = model.bert(uncond_input.input_ids.to(model.device))[0]
    
    text_input = model.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt")
    text_embeddings = model.bert(text_input.input_ids.to(model.device))[0]
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    context = torch.cat([uncond_embeddings, text_embeddings])
    
    model.scheduler.set_timesteps(num_inference_steps)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale)
    
    image = latent2image(model.vqvae, latents)
   
    return image, latent



# @torch.no_grad()
# def text2image_ldm_stable(
#     model,
#     prompt: List[str],
#     controller,
#     num_inference_steps: int = 50,
#     guidance_scale: float = 7.5,
#     generator: Optional[torch.Generator] = None,
#     latent: Optional[torch.FloatTensor] = None,
#     low_resource: bool = False,
# ):
#     register_attention_control(model, controller)
#     height = width = 512
#     batch_size = len(prompt)


#     text_input = model.tokenizer(
#         prompt,
#         padding="max_length",
#         max_length=model.tokenizer.model_max_length,
#         truncation=True,
#         return_tensors="pt",
#     )
#     text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
# #     print(text_embeddings.shape)
#     max_length = text_input.input_ids.shape[-1]

#     # negative_prompt = (
#     #     "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face,"
#     #     + "out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy,"
#     #     + "watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face : -2"
#     # )

#     init_step_controller = controller
#     last_step_controller = controller

#     uncond_input = model.tokenizer(
#         [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
#     )
#     uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    
#     context = [uncond_embeddings, text_embeddings]
#     if not low_resource:
#         context = torch.cat(context)
#     latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    
#     # set timesteps
#     extra_set_kwargs = {"offset": 1}
#     model.scheduler.set_timesteps(num_inference_steps)
#     for t in tqdm(model.scheduler.timesteps):
#         latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource)
    
#     image = latent2image(model.vae, latents)
  
#     return image, latent
from torchvision import transforms
def preprocess_image(image_path, img_size=(512, 512)):
    # 이미지 로드
    image = Image.open(image_path).convert('RGB')
    
    # 이미지 전처리 파이프라인 정의
    transform = transforms.Compose([
        transforms.Resize(img_size),  # 이미지 크기 조정
        transforms.ToTensor(),  # 이미지 Tensor로 변환
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 이미지 정규화
    ])
    
    # 전처리 적용
    image = transform(image)
    image = (image - image.min()) / (image.max() - image.min())
    # 배치 차원 추가 (모델에 입력할 때 배치 차원이 필요함)
    image = image.unsqueeze(0)
    
    return image

# 예시: 이미지 경로를 입력으로 사용

def diff_controller(a , b):
    result = [torch.abs(t1 - t2) for t1, t2 in zip(a, b)]
    return result

def agg_controller(a , b):
    result = [t1 + t2 for t1, t2 in zip(a, b)]
    return result

def extract_controller_item(a):
    list_cpu = [tensor.to('cpu') for tensor in a]
    return list_cpu

def diff_controller_aggregation(diff_mid_agg , diff_up_agg , target_location):
    # mid map aggregation
    mid_map = diff_mid_agg[0][:,:,target_location]
    mid_map = torch.sum(mid_map , dim=0)
    mid_map_norm = (mid_map - mid_map.min()) / (mid_map.max() - mid_map.min())
    mid_map_norm_reshape = cv2.resize(mid_map_norm.reshape(8,8).numpy(), (64, 64), interpolation=cv2.INTER_LINEAR)

    # up map aggregation
    up_map_16_1 = diff_up_agg[0][:,:,4]
    up_map_16_1 = torch.sum(up_map_16_1 , dim=0)
    up_map_16_2 = diff_up_agg[1][:,:,4]
    up_map_16_2 = torch.sum(up_map_16_2 , dim=0)
    up_map_16_3 = diff_up_agg[2][:,:,4]
    up_map_16_3 = torch.sum(up_map_16_3 , dim=0)

    up_map_16 = up_map_16_1 + up_map_16_2 + up_map_16_3
    up_map_16_norm = (up_map_16 - up_map_16.min()) / (up_map_16.max() - up_map_16.min())
    up_map_16_norm_reshape = cv2.resize(up_map_16_norm.reshape(16,16).numpy(), (64, 64), interpolation=cv2.INTER_LINEAR)

    up_map_32_1 = diff_up_agg[3][:,:,4]
    up_map_32_1 = torch.sum(up_map_32_1 , dim=0)
    up_map_32_2 = diff_up_agg[4][:,:,4]
    up_map_32_2 = torch.sum(up_map_32_2 , dim=0)
    up_map_32_3 = diff_up_agg[5][:,:,4]
    up_map_32_3 = torch.sum(up_map_32_3 , dim=0)

    up_map_32 = up_map_32_1 + up_map_32_2 + up_map_32_3
    up_map_32_norm = (up_map_32 - up_map_32.min()) / (up_map_32.max() - up_map_32.min())
    up_map_32_norm_reshape = cv2.resize(up_map_32_norm.reshape(32,32).numpy(), (64, 64), interpolation=cv2.INTER_LINEAR)

    agg_map = mid_map_norm_reshape + up_map_16_norm_reshape + up_map_32_norm_reshape
    agg_map = torch.tensor(agg_map)
    agg_map_norm = (agg_map - agg_map.min()) / (agg_map.max() - agg_map.min())
    return agg_map_norm

def initialize_list_to_zero(tensor_list):
    return [torch.zeros_like(tensor) for tensor in tensor_list]






@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt: List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    low_resource: bool = False,
    input_img_path = None,
    target_location = None
):
    register_attention_control(model, controller)
    height = width = 512
    batch_size = len(prompt)


    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    eot_location = torch.where(text_input['input_ids'] == 49407)[1][0].item()
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    # eot zero padding
    # text_embeddings[0][eot_location+1:] = text_embeddings[0][eot_location-1]
#     print(text_embeddings.shape)
    max_length = text_input.input_ids.shape[-1]

    negative_prompt = (
        "background"
    )

    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    
    context = [uncond_embeddings, text_embeddings]
    del uncond_embeddings , text_embeddings
    torch.cuda.empty_cache()
    if not low_resource:
        context = torch.cat(context)
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    
    # set timesteps
    extra_set_kwargs = {"offset": 1}
    model.scheduler.set_timesteps(num_inference_steps)
    input_img = preprocess_image(input_img_path)
    input_img = input_img * 2 - 1
    posterior = model.vae.encode(input_img.to('cuda'))['latent_dist'].sample(generator = torch.Generator().manual_seed(0)) * 0.18215
    last_controller = None
    for i , t in enumerate(tqdm(model.scheduler.timesteps)):
        # if i == 0:
        #     init_latents = latents
        #     dummy_latents = diffusion_step(model, init_step_controller, init_latents, context, t, guidance_scale, low_resource)
        #     model.scheduler.counter = i
        # if i == num_inference_steps:
        #     last_latents = latents
        #     dummy_latents = diffusion_step(model, last_step_controller, last_latents, context, t, guidance_scale, low_resource)
        #     model.scheduler.counter = i
        # t = 999 - t
        # if i == 0:
        #     latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource , posterior)
        #     image = latent2image(model.vae, latents)
        #     before_up_agg = extract_controller_item(controller.attention_store['up_cross'])
        #     before_mid_agg = extract_controller_item(controller.attention_store['mid_cross'] )
        #     before_down_agg = extract_controller_item(controller.attention_store['down_cross'])
        # if i == 10:
            # controller.attention_store['up_cross'] = initialize_list_to_zero(controller.attention_store['up_cross'])
            # controller.attention_store['mid_cross'] = initialize_list_to_zero(controller.attention_store['mid_cross'])
            # controller.attention_store['down_cross'] = initialize_list_to_zero(controller.attention_store['down_cross'])
            # latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource , posterior)
            # image = latent2image(model.vae, latents)
            # ca_8 = torch.sum(controller.attention_store['mid_cross'][0][:,:,3] , dim=0)
            # ca_8_resize = cv2.resize(ca_8.cpu().numpy() , (16,16))
            # ca_8_resize = torch.tensor(ca_8_resize.reshape(16*16)).to(latents.device)
            # before_cross_attn_map = torch.sum(controller.attention_store['down_cross'][5][:,:,target_location] , dim=0).to(latents.device) + torch.sum(controller.attention_store['down_cross'][4][:,:,target_location] , dim=0).to(latents.device) + torch.sum(controller.attention_store['up_cross'][0][:,:,target_location] , dim=0).to(latents.device)
            # + torch.sum(controller.attention_store['up_cross'][1][:,:,target_location] , dim=0).to(latents.device) + torch.sum(controller.attention_store['up_cross'][2][:,:,target_location] , dim=0).to(latents.device) + ca_8_resize
        
        if i == 60:
            controller.attention_store['up_cross'] = initialize_list_to_zero(controller.attention_store['up_cross'])
            controller.attention_store['mid_cross'] = initialize_list_to_zero(controller.attention_store['mid_cross'])
            controller.attention_store['down_cross'] = initialize_list_to_zero(controller.attention_store['down_cross'])
            latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource , posterior)
            image = latent2image(model.vae, latents)
            ca_8 = torch.sum(controller.attention_store['mid_cross'][0][:,:,target_location] , dim=0)
            ca_8_resize = cv2.resize(ca_8.cpu().numpy() , (16,16))
            ca_8_resize = torch.tensor(ca_8_resize.reshape(16*16)).to(latents.device)
            cross_attn_map = torch.sum(controller.attention_store['down_cross'][5][:,:,target_location] , dim=0).to(latents.device) + torch.sum(controller.attention_store['down_cross'][4][:,:,target_location] , dim=0).to(latents.device) + torch.sum(controller.attention_store['up_cross'][0][:,:,target_location] , dim=0).to(latents.device)
            + torch.sum(controller.attention_store['up_cross'][1][:,:,target_location] , dim=0).to(latents.device) + torch.sum(controller.attention_store['up_cross'][2][:,:,target_location] , dim=0).to(latents.device) 
            # cross_attn_map = torch.abs(cross_attn_map - before_cross_attn_map)
            # del before_cross_attn_map
            # cross_attn_map = torch.sum(controller.attention_store['down_cross'][5][:,:,3],dim=0) + torch.sum(controller.attention_store['down_cross'][4][:,:,3],dim=0) + ca_8_resize
            # cross_attn_map = torch.sum(cross_attn_map[:,:,3] , dim=0)

        elif i == 0:
            # before_up_agg = extract_controller_item(controller.attention_store['up_cross'])
            # before_mid_agg = extract_controller_item(controller.attention_store['mid_cross'] )
            # before_down_agg = extract_controller_item(controller.attention_store['down_cross'])
            latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource , posterior)
            image = latent2image(model.vae, latents)
            
        elif i == 70:
            latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource , posterior)
            image = latent2image(model.vae, latents)
        # else:
        #     latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource , posterior)
        #     image = latent2image(model.vae, latents)
            # latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource , posterior)
            # image = latent2image(model.vae, latents)
        # Diff Aggregation    
        # if i == 55:
        #     before_up_agg = extract_controller_item(controller.attention_store['up_cross'])
        #     before_mid_agg = extract_controller_item(controller.attention_store['mid_cross'] )
        #     before_down_agg = extract_controller_item(controller.attention_store['down_cross'])
        #     latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource , posterior)
        #     image = latent2image(model.vae, latents)
        #     cur_up_state = diff_controller(extract_controller_item(controller.attention_store['up_cross']) , before_up_agg)
        #     cur_mid_state = diff_controller(extract_controller_item(controller.attention_store['mid_cross']) , before_mid_agg)
        #     cur_down_state = diff_controller(extract_controller_item(controller.attention_store['down_cross']) , before_down_agg)
        #     ca_8 = torch.sum(cur_mid_state[0][:,:,target_location] , dim=0)
        #     ca_8_resize = cv2.resize(ca_8.cpu().numpy() , (16,16))
        #     ca_8_resize = torch.tensor(ca_8_resize.reshape(16*16)).to(latents.device)
        #     cross_attn_map = torch.sum(cur_down_state[4][:,:,target_location] , dim=0).to(latents.device) + torch.sum(cur_down_state[5][:,:,target_location] , dim=0).to(latents.device) + torch.sum(cur_up_state[0][:,:,target_location] , dim=0).to(latents.device)
        #     + torch.sum(cur_up_state[1][:,:,target_location] , dim=0).to(latents.device) + torch.sum(cur_up_state[2][:,:,target_location] , dim=0).to(latents.device) + ca_8_resize
        #     del before_up_agg , before_mid_agg , before_down_agg , cur_up_state , cur_mid_state , cur_down_state
            
            
            

        # elif i == 21:
        #     before_up_state = cur_up_state                    
        #     before_mid_state = cur_mid_state                    
        #     before_down_state = cur_down_state                    
        #     latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource , posterior)
        #     image = latent2image(model.vae, latents)
        #     cur_up_state = diff_controller(extract_controller_item(controller.attention_store['up_cross']) , before_up_agg)
        #     cur_mid_state = diff_controller(extract_controller_item(controller.attention_store['mid_cross']) , before_mid_agg)
        #     cur_down_state = diff_controller(extract_controller_item(controller.attention_store['down_cross']) , before_down_agg)
        #     before_up_agg = extract_controller_item(controller.attention_store['up_cross'])
        #     before_mid_agg = extract_controller_item(controller.attention_store['mid_cross'] )
        #     before_down_agg = extract_controller_item(controller.attention_store['down_cross'])
        #     diff_up_agg =  diff_controller(cur_up_state , before_up_state)
        #     diff_mid_agg =  diff_controller(cur_mid_state , before_mid_state)
        #     diff_down_agg =  diff_controller(cur_down_state , before_down_state)
        # elif (i > 21):
        #     before_up_state = cur_up_state                    
        #     before_mid_state = cur_mid_state                    
        #     before_down_state = cur_down_state                    
        #     latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource , posterior)
        #     image = latent2image(model.vae, latents)
        #     cur_up_state = diff_controller(extract_controller_item(controller.attention_store['up_cross']) , before_up_agg)
        #     cur_mid_state = diff_controller(extract_controller_item(controller.attention_store['mid_cross']) , before_mid_agg)
        #     cur_down_state = diff_controller(extract_controller_item(controller.attention_store['down_cross']) , before_down_agg)
        #     before_up_agg = extract_controller_item(controller.attention_store['up_cross'])
        #     before_mid_agg = extract_controller_item(controller.attention_store['mid_cross'] )
        #     before_down_agg = extract_controller_item(controller.attention_store['down_cross'])
        #     diff_up_agg = agg_controller(diff_up_agg , diff_controller(cur_up_state , before_up_state))
        #     diff_mid_agg = agg_controller(diff_mid_agg , diff_controller(cur_mid_state , before_mid_state))
        #     diff_down_agg = agg_controller(diff_down_agg , diff_controller(cur_down_state , before_down_state))
        # else:
        #     latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource , posterior)
        #     image = latent2image(model.vae, latents)
    # diff_up_agg = agg_controller(before_down_agg , agg_controller(before_up_agg , before_mid_agg))        
    # diff_agg_map = diff_controller_aggregation(diff_mid_agg , diff_up_agg , target_location=4)
    diff_agg_map = None
    image = latent2image(model.vae, latents)
    # del latents , uncond_embeddings , text_embeddings, dummy_latents 
    torch.cuda.empty_cache()
    # return image, latent, init_step_controller, last_step_controller
    return image, latent, diff_agg_map , cross_attn_map


@torch.no_grad()
def text2image_ldm_stable_seediff(
    model,
    prompt: List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    low_resource: bool = False,
):
    register_attention_control(model, controller)
    height = width = 512
    batch_size = len(prompt)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    
    context = [uncond_embeddings, text_embeddings]
    if not low_resource:
        context = torch.cat(context)
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    
    # set timesteps
    extra_set_kwargs = {"offset": 1}
    model.scheduler.set_timesteps(num_inference_steps)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step_seediff(model, controller, latents, context, t, guidance_scale, low_resource)
    
    image = latent2image(model.vae, latents)
  
    return image, latent





import time

def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out
        
        def reshape_heads_to_batch_dim(self, tensor):
            batch_size, seq_len, dim = tensor.shape
            head_size = self.heads
            tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
            tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
            return tensor

        def reshape_batch_dim_to_heads(self, tensor):
            batch_size, seq_len, dim = tensor.shape
            head_size = self.heads
            tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
            tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
            return tensor

        # def forward(x, context=None, mask=None):
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
            x = hidden_states
            context = encoder_hidden_states
            mask = attention_mask
            
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            
            k = self.to_k(context)
            v = self.to_v(context)
            
            q = reshape_heads_to_batch_dim(self,q)
            k = reshape_heads_to_batch_dim(self,k)
            v = reshape_heads_to_batch_dim(self,v)
            # 터짐
            del encoder_hidden_states
            time.sleep(0.001)
            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
            del q, k 
            torch.cuda.empty_cache()

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            del sim , x, hidden_states 
            torch.cuda.empty_cache()

            attn = controller(attn, is_cross, place_in_unet)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            
            out = reshape_batch_dim_to_heads(self,out)
            
            res = to_out(out)
            del attn , v , out
            torch.cuda.empty_cache()
            return res

        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet, module_name=None):
        # if net_.__class__.__name__ == 'CrossAttention':
        #     net_.forward = ca_forward(net_, place_in_unet)
        #     return count + 1
        if module_name in ["attn1", "attn2"]:
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for k,net__ in net_.named_children():
                count = register_recr(net__, count, place_in_unet, module_name = k)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count
    

# def register_attention_control(model, controller):
    
    
#     def ca_forward(self, place_in_unet):
        
# #         def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
# #             x = hidden_states
# #             context = encoder_hidden_states
# #             mask = attention_mask
            
#         def forward(x, context=None, mask=None):
#             batch_size, sequence_length, dim = x.shape
#             h = self.heads
#             q = self.to_q(x)
#             is_cross = context is not None
#             context = context if is_cross else x
#             k = self.to_k(context)
#             v = self.to_v(context)
#             q = self.reshape_heads_to_batch_dim(q)
#             k = self.reshape_heads_to_batch_dim(k)
#             v = self.reshape_heads_to_batch_dim(v)

#             sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

#             if mask is not None:
#                 mask = mask.reshape(batch_size, -1)
#                 max_neg_value = -torch.finfo(sim.dtype).max
#                 mask = mask[:, None, :].repeat(h, 1, 1)
#                 sim.masked_fill_(~mask, max_neg_value)

#             # attention, what we cannot get enough of
#             attn = sim.softmax(dim=-1)
            
#             attn = controller(attn, is_cross, place_in_unet)
#             out = torch.einsum("b i j, b j d -> b i d", attn, v)
#             out = self.reshape_batch_dim_to_heads(out)
#             out = self.to_out(out)
#             return out

#         return forward

#     def register_recr(net_, count, place_in_unet):
#         if net_.__class__.__name__ == 'CrossAttention':
#             net_.forward = ca_forward(net_, place_in_unet)
#             return count + 1
#         elif hasattr(net_, 'children'):
#             for net__ in net_.children():
#                 count = register_recr(net__, count, place_in_unet)
#         return count

#     cross_att_count = 0
#     sub_nets = model.unet.named_children()
#     for net in sub_nets:
# #         print(net)
#         if "down" in net[0]:
#             cross_att_count += register_recr(net[1], 0, "down")
#         elif "up" in net[0]:
#             cross_att_count += register_recr(net[1], 0, "up")
#         elif "mid" in net[0]:
#             cross_att_count += register_recr(net[1], 0, "mid")
#     controller.num_att_layers = cross_att_count

    
def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def update_alpha_time_word(alpha, bounds: Union[float, Tuple[float, float]], prompt_ind: int, word_inds: Optional[torch.Tensor]=None):
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[: start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                                   tokenizer, max_num_words=77):
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"],
                                                  i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
             inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
             for i, ind in enumerate(inds):
                 if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words) # time, batch, heads, pixels, words
    return alpha_time_words



def latent2diff(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
#     print(image.shape)
    
#     avgPool = nn.AvgPool2d(2)  #4*4的窗口，步长为4的平均池化
#     image_1 = avgPool(image)
    
    diff_image = image[-1] - image[0]
    diff_mean = diff_image.abs().mean(dim=0)
    
    diff_norm = (diff_mean - diff_mean.min())/(diff_mean.max() - diff_mean.min())
    # diff_image = (diff_normed > 0.5).float()
    diff_norm = diff_norm.cpu()


    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)

    return image, diff_norm

@torch.no_grad()
def text2image_ldm_diff(
        model,
        prompt: List[str],
        controller,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
        latent: Optional[torch.FloatTensor] = None,
        low_resource: bool = False,
):
    register_attention_control(model, controller)
    height = width = 512
    batch_size = len(prompt)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    context = [uncond_embeddings, text_embeddings]
    if not low_resource:
        context = torch.cat(context)

    latent, latents = init_latent(latent, model, height, width, generator, batch_size)


    # set timesteps
    extra_set_kwargs = {"offset": 1}
    model.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)


    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource)

    image, diff_norm = latent2diff(model.vae, latents)

    return image, diff_norm, latent

@torch.no_grad()
def text2image_ldm_difflatent(
        model,
        prompt: List[str],
        controller,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
        latent: Optional[torch.FloatTensor] = None,
        low_resource: bool = False,
):

    register_attention_control(model, controller)
    height = width = 512
    batch_size = len(prompt)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    
    diff = True
    if diff:
        text_embeddings = text_embeddings[0][None]
    #     delta = torch.randn(text_embeddings.shape[-1])
        delta = torch.ones(text_embeddings.shape[-1]) 

        delta = delta[None].to(text_embeddings)
        noise_text_embeddings = copy.deepcopy(text_embeddings)
        # 1 77 768 noise_text_embeddings
        noise_text_embeddings[:, 4] += delta

        context = [uncond_embeddings, text_embeddings,noise_text_embeddings]
    
    else:
        context = [uncond_embeddings, text_embeddings]
        
    if not low_resource:
        context = torch.cat(context)

    latent, latents = init_latent(latent, model, height, width, generator, batch_size)

    # set timesteps
    extra_set_kwargs = {"offset": 1}
    model.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource)

    image, diff_norm = latent2diff(model.vae, latents)

    return image, diff_norm, latent