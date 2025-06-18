#%%
import sys
sys.path.append("/root/code/IP-Adapter")
import cv2
import torch
import os
current_path = os.getcwd()
print("p",current_path)
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL, UNet2DConditionModel
from PIL import Image

from ip_adapter.ip_adapter_faceid_multi_pose import *

def create_bbox_mask(image_size, bbox):
    mask = torch.zeros(image_size)
    x1, y1, x2, y2 = bbox

    mask[y1:y2+1,x1:x2+1] = 1

    return mask


from tqdm import tqdm
import json
import numpy as np
if __name__ == "__main__":
    # arvg = sys.argv
    # if len(arvg) == 1:
    #     print("Please input the image path")
    #     sys.exit(0)
    # ip_ckpts = arvg[1:]

    prompt = "a man and a man in suits posing for a picture with one of them holding a watch"
    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"
    num_samples = 4
    all_iamges = []
    output_paths = []

    image_root_path = "/root/data/fastcomposer/fastcomposer/ffhq3/"
    image_id = "000062063"#"000030334"#"000052852"#"000057365"#"000005648"#"000030334"#"000029937"#000030334"#"000030305"#"000066481"#
    chunk = image_id[:5]
    image_path = os.path.join(image_root_path, chunk, image_id, image_id + ".jpg")
    info_path = os.path.join(image_root_path, chunk, image_id, image_id + ".json")

    with open(info_path, "r") as f:
        info_dict = json.load(f)#caption,xy

    bbox = info_dict["xy"][0]
    print(bbox)
    prompt = info_dict['caption']
    #prompt = "a man and a man in swimsuits standing next to each other on a river"
    info_dict = info_dict["pose"][0]
    face_num = min(3,len(info_dict))#########



    xy_embeds = []
    face_embeds = []
    face_masks = []
    bbox_list = []
    for idx in range(face_num):
        xy_embed = torch.tensor(info_dict[str(idx)], dtype=torch.float32) / 511.0
        xy_embed = xy_embed.reshape(xy_embed.shape[0]*xy_embed.shape[1])
        face_path = os.path.join(image_root_path, chunk, image_id, str(idx) + ".npy")
        face_id_embed = np.load(face_path,allow_pickle=True)
        face_id_embed = torch.from_numpy(face_id_embed)
        
        bbox_list.append(bbox[str(idx)])
        face_mask = torch.ones((64,64))
        face_masks.append(torch.ones(64,64))
        for qq in range(0,3):
            face_masks.append(face_mask)

        xy_embeds.append(xy_embed)
        face_embeds.append(face_id_embed)#######

    while len(face_embeds) < 3:
        face_embeds.append(torch.full_like(face_embeds[0], 0))
        xy_embeds.append(torch.full_like(xy_embeds[0], -1))#########padding -1
        for qq in range(0,4):
            face_masks.append(torch.zeros(64,64))

    xy_embeds = torch.stack(xy_embeds).unsqueeze(0)
    print(xy_embeds.shape)
    print("xy",xy_embeds)

    mask = torch.zeros(12)#
    mask[:2*4] = 1
    mask = mask.unsqueeze(0)
    mask = mask.repeat(num_samples,1)
    print("mask",mask)


    bbox_list = [bbox_list]*num_samples*2
    face_masks = torch.stack(face_masks).unsqueeze(0)
    print(face_masks.shape)
    face_masks = face_masks.repeat(num_samples*2,1,1,1)#
    face_embeds = torch.stack(face_embeds).unsqueeze(0)
   
    img_file_paths = ["/root/code/transformers_llama/notes/person-img/tmoxi.npy","/root/code/transformers_llama/notes/person-img/anni.npy"]
    
    face_embeds = []
    for file in img_file_paths:
        face_id_embed = np.load(file,allow_pickle=True)
        face_id_embed = torch.from_numpy(face_id_embed)
        face_embeds.append(face_id_embed)#######
    while len(face_embeds) < 3:
        face_embeds.append(torch.full_like(face_embeds[0], 0))

    face_embeds = torch.stack(face_embeds).unsqueeze(0)
    kwargs = {"mask":mask}
    cross_attention_kwargs = {"face_masks":face_masks, "bbox_list":bbox_list}

    base_model_path = "/root/base_model/stable-diffusion-v1-5"
    vae_model_path = "/root/base_model/sd-vae-ft-mse/"
    device = "cuda"

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        # vae=vae,
        # unet=unet,
        feature_extractor=None,
        safety_checker=None
    )
    go = False
    # path 
    if 1:
        print("批量跑")
        dir = "/root/code/IP-Adapter/multiface/logs/718-2e4_tuned_12_4token_local_nopose_faceidnocros_nodrop_5_1.0_noise"#your ckpt file
        ip_ckpts = [os.path.join(dir, f) for f in os.listdir(dir)]
        ip_ckpts = sorted(ip_ckpts, key=lambda x: int(x.split("/")[-1].split("-")[-1]),reverse=True)#

    for ip_ckpt in tqdm(ip_ckpts):
        
        images = []
        if not ip_ckpt.endswith(".bin"):
            ip_ckpt = os.path.join(ip_ckpt, "ip_adapter.bin")
        if not os.path.exists(ip_ckpt):
            print(f"{ip_ckpt} not exists")
            continue
        # load ip-adapter
        ip_model = IPAdapterFaceID_p3(pipe, ip_ckpt, device,num_tokens=4)
    
        images += ip_model.generate(
                prompt=prompt, negative_prompt=negative_prompt, xy_embeds=xy_embeds, c_embeds=face_embeds, num_samples=num_samples, width=512, height=512, num_inference_steps=30, \
                    cross_attention_kwargs=cross_attention_kwargs,seed=2023)#,**kwargs)#
        # 拼接
        result = Image.new('RGB', (images[0].width * len(images), images[0].width),(255, 255, 255))
        for i, img in enumerate(images):
            result.paste(img, (i * img.width+i*2, 0))
        all_iamges.append(result)
        output_path = ip_ckpt.replace(".bin", "_test.jpg")
        print("save: ", output_path)
        result.save(output_path)
        output_paths.append(output_path)
    for output_path in output_paths:
        print(output_path)
