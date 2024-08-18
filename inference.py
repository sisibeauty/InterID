import sys
import cv2
import torch
import our_pipeline_stable_diffusion
from our_pipeline_stable_diffusion import StableDiffusionPipeline_our as StableDiffusionPipeline
from diffusers import DDIMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import cv2
from PIL import Image
import os
import glob
import json
import re
import matplotlib.pyplot as plt
from torchvision import transforms
from diffusers.utils import load_image
from tqdm import tqdm
import math
import PIL
img_resolution = 512
img_transform = transforms.Compose([
            transforms.Resize(img_resolution, interpolation=Image.BILINEAR),
            transforms.CenterCrop(img_resolution),
        ])

#load pose_extractor
device = torch.device("cuda")
model_name = "pose_extractor_path"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
#load sd
base_model_path = "sd1.5path"
vae_model_path = "vae_model_path"
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
    vae=vae,
    feature_extractor=None,
    safety_checker=None
)
from InterID import *
interid_ckpt = "interid_ckpt_path"
interid_model = InterID_p3(pipe, ip_ckpt, device,num_tokens=4)
prompt = "a man hugs a woman in the park"
face_img = ["man.jpg","woman.jpg"]
face_embedding = []
for face in face_img:
    face_image = load_image(face)
    image =  cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR)
    faces = app.get(image)
    face_info = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]),reverse=True)[0]
    id = face_info.normed_embedding
    id = torch.from_numpy(id)
    face_embedding.append(id)

pose_result = []
inputs = tokenizer(prompt+"\npose (2ID):", return_tensors="pt")
inputs.to(device)
sentence = model.generate(
inputs.input_ids,do_sample=True,max_length=256,)       
decoded_output = tokenizer.decode(sentence[0])
target = "ID):"
start_index = decoded_output.find(target)
end_index = start_index + len(target)
xys = decoded_output[end_index:]
numbers = re.findall(r'\d+', xys)
pose_result = [list(map(float, numbers[i:i+2])) for i in range(0, len(numbers), 2)]
pose_result = [pose_result[i:i+5] for i in range(0, len(pose_result), 5)]

#generate
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"
num_images_per_prompt = 4   
xy_embeds = []
for idx in range(2):
    xy_embed = torch.tensor(pose_result[idx], dtype=torch.float32) / 511.0
    xy_embed = xy_embed.reshape(xy_embed.shape[0]*xy_embed.shape[1])
    xy_embeds.append(xy_embed)
xy_embeds = torch.stack(xy_embeds).unsqueeze(0)
face_embeds = torch.stack(face_embedding).unsqueeze(0)
output_images = interid_model.generate(
    prompt=prompt, negative_prompt=negative_prompt, xy_embeds=xy_embeds, c_embeds=face_embeds, \
    scale=1,guidance_scale=3,num_samples=num_images_per_prompt, width=512, height=512, num_inference_steps=50)
output_images.save('results.jpg')



