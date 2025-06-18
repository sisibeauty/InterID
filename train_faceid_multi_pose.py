import sys
sys.path.append("/root/code/IP-Adapter")
is_multi = True
import os
import random
import argparse
from pathlib import Path
import json
import itertools
import time

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

if is_multi:
    from ip_adapter.ip_adapter_faceid_multi_pose import MLPProjModel # 这块还没改,%%gail 
else:
    from ip_adapter.ip_adapter_faceid_cricle import MLPProjModel

from ip_adapter.utils import is_torch2_available
from ip_adapter.attention_processor_faceid_maskq_1 import LoRAAttnProcessor, LoRAIPAttnProcessor_p3
from tqdm import tqdm
import numpy as np
from ip_adapter.utils_train import register_cross_attention_hook, get_net_attn_map, attnmaps2images, get_raw_attn_map, \
    BalancedL1Loss,get_object_localization_loss_for_one_layer, clear_cross_attention_scores
import matplotlib.pyplot as plt

def create_bbox_mask(image_size, bbox):
    mask = torch.ones(image_size)
    x1, y1, x2, y2 = bbox

    # 将bbox的四个点位置设为1
    # mask[y1, x1:x2+1] = 1  # 上边界
    # mask[y2, x1:x2+1] = 1  # 下边界
    # mask[y1:y2+1, x1] = 1  # 左边界
    # mask[y1:y2+1, x2] = 1  # 右边界
    mask[y1:y2+1,x1:x2+1] = 1

    return mask

#fastcomposer dataset face+bbox pos
# Dataset
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer, size=512, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05,xy_drop_rate=0.05, 
                 image_root_path="/root/data/fastcomposer/fastcomposer/ffhq3",split="all"):
        super().__init__()

        self.tokenizer = tokenizer
        self.size = size
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.xy_drop_rate = xy_drop_rate##################
        self.image_root_path = image_root_path
        # files = os.listdir(image_root_path)#[:20]
        # self.data = []

        if split == "all":
            image_ids_path = os.path.join(image_root_path, "image_ids.txt")
        elif split == "train":
            image_ids_path = os.path.join(image_root_path, "image_ids-tuned.txt")##image_ids.txt
        elif split == "test":
            image_ids_path = os.path.join(image_root_path, "image_ids.txt")
        else:
            raise ValueError(f"Unknown split {split}")

        with open(image_ids_path, "r") as f:
            self.image_ids = f.read().splitlines()
            
        print("data dir:", image_root_path)
        print("file num:", len(self.image_ids))

        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.segmap_resize = transforms.Resize(
            64,###
            interpolation=transforms.InterpolationMode.NEAREST,
        )
        self.count = 0
        
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        chunk = image_id[:5]
        image_path = os.path.join(self.image_root_path, chunk, image_id, image_id + ".jpg")
        info_path = os.path.join(self.image_root_path, chunk, image_id, image_id + "_.json")##
        objmsk_path = os.path.join(self.image_root_path, chunk, image_id, 'mask')

        with open(info_path, "r") as f:
            info_dict = json.load(f)#caption,xy

       
        text = info_dict['caption']
    
        
        # read image
        raw_image = Image.open(image_path)
        image = self.transform(raw_image.convert("RGB"))

        xy_embeds = []
        face_embeds = []
        face_masks = []
        bbox_list = []
        bbox = info_dict["xy"][0]
        info_dict = info_dict["pose"][0]
        face_num = len(info_dict)# - 3#1,2
        face_num = min(face_num,3)##数据集最多有6张

        facem_list = []
        objmsk_list = []
        # for idx in range(face_num):
        #     msk_path = os.path.join(objmsk_path,str(idx)+'.npy')
        #     #objmsk_path = "/root/data/fastcomposer/fastcomposer/ffhq3/00010/000100000/mask/0.npy"
        #     try:
        #         objmsk = np.load(msk_path)#.float()
        #     except:
        #         print("error:", msk_path)
        #     facem_list.append(objmsk)

        # for ff in facem_list:
        #     objmsk = torch.from_numpy(ff).unsqueeze(0)
        #     #print(objmsk.shape)
        #     objmsk = self.segmap_resize(objmsk)
        #     objmsk = objmsk.squeeze(0)
        #     #objmsk_list.append(bgmsk)
        #     objmsk_list.append(torch.zeros_like(objmsk))
        #     objmsk_list += [objmsk]*3

        # while len(objmsk_list) < 12:
        #     objmsk_list += [torch.zeros_like(objmsk_list[0])]*4
        # # # 创建一个子图网格，每行显示一个掩码张量
        # rows = 1
        # cols = len(objmsk_list)

        # fig, axes = plt.subplots(rows, cols, figsize=(3, 3 * rows))

        # # 遍历列表中的张量并可视化
        # for i, tensor in enumerate(objmsk_list):
        #     # 将张量转换为NumPy数组
        #     mask_array = tensor.numpy()

        #     # 显示掩码图像
        #     axes[i].imshow(mask_array, cmap='gray')
        #     axes[i].axis('off')

        # plt.tight_layout()
        # plt.savefig('/root/temp/mask'+str(self.count)+'.png', dpi=300)
        # self.count += 1
        #exit(0)


        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1
        # get text and tokenize
        text_input_ids = self.tokenizer(
                text,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids
        for idx in range(face_num):
            xy_embed = torch.tensor(info_dict[str(idx)], dtype=torch.float32) / 511.0
            xy_embed = xy_embed.reshape(xy_embed.shape[0]*xy_embed.shape[1])
            #c_embed = torch.tensor(c, dtype=torch.float32) / 255.0
            face_path = os.path.join(self.image_root_path, chunk, image_id, str(idx) + ".npy")
            face_id_embed = np.load(face_path,allow_pickle=True)
            face_id_embed = torch.from_numpy(face_id_embed)
            bbox1 = [min(63,int((coord / 511) * 63 * 1.25)) for coord in bbox[str(idx)]]
            bbox1 = [max(0,coord) for coord in bbox1]
            bbox_list.append(bbox[str(idx)])
            face_mask = create_bbox_mask((64,64), bbox1)
            #face_mask = torch.ones(64,64)
            face_masks.append(torch.ones(64,64))#前面token都为1，后面token都为0
            for qq in range(0,3):
                face_masks.append(face_mask)

            if drop_image_embed:
                face_id_embed = torch.zeros_like(face_id_embed)

            
            
            
            xy_embeds.append(xy_embed)
            face_embeds.append(face_id_embed)#######


        while len(face_embeds) < 3:
            try:
                face_embeds.append(torch.full_like(face_embeds[0], 0))
                xy_embeds.append(torch.full_like(xy_embeds[0], -1))#########padding -1
                for qq in range(0,4):
                    face_masks.append(torch.zeros(64,64))
            except:
                print(image_id,info_dict,face_num)
           
        
        face_embeds = torch.stack(face_embeds)
        xy_embeds = torch.stack(xy_embeds)
        face_masks = torch.stack(face_masks)

        attn_mask = torch.zeros(12)###20?
        attn_mask[:face_num*4] = 1
        
        return {
                "image": image,
                "text_input_ids": text_input_ids,
                "drop_image_embed": drop_image_embed,
                "xy_embed": xy_embeds,
                "c_embed": face_embeds,
                "attn_mask":attn_mask,
                "face_mask":face_masks,
                "bbox_list":bbox_list,
                #"objmsks":objmsks,
            }


    def __len__(self):
        return len(self.image_ids)
    

def collate_fn_multi(data):
    images = torch.stack([example["image"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    xy_embed = torch.stack([example["xy_embed"] for example in data])
    c_embed = torch.stack([example["c_embed"] for example in data])
    drop_image_embeds = [example["drop_image_embed"] for example in data]
    attn_mask = torch.stack([example["attn_mask"] for example in data])
    face_mask = torch.stack([example["face_mask"] for example in data])
    bbox_list = [example["bbox_list"] for example in data]
    return {
        "images": images,
        "text_input_ids": text_input_ids,
        "xy_embed": xy_embed,
        "c_embed": c_embed,
        "drop_image_embeds": drop_image_embeds,
        "attn_mask":attn_mask,
        "face_mask":face_mask,
        "bbox_list":bbox_list,
    }
    

def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    xy_embed = torch.stack([example["xy_embed"] for example in data])
    c_embed = torch.stack([example["c_embed"] for example in data])
    drop_image_embeds = [example["drop_image_embed"] for example in data]

    return {
        "images": images,
        "text_input_ids": text_input_ids,
        "xy_embed": xy_embed,
        "c_embed": c_embed,
        "drop_image_embeds": drop_image_embeds
    }
    


class IPAdapter(torch.nn.Module):
    """IP-Adapter"""
    def __init__(self, unet, image_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules
        # ckpt_path = "/root/code/IP-Adapter/multiface/logs/607_12_4token_pose_faceidnocros_nodrop_5_1.0_noise/Epoch-28-checkpoint-236000/ip_adapter.bin"
        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, xy_embeds, c_embeds,cross_attention_kwargs):
        
        b, n, c = xy_embeds.shape
        xy_embeds = xy_embeds.reshape(b*n,c)
        b, n, c = c_embeds.shape
        c_embeds = c_embeds.reshape(b*n,c)########

        ip_tokens = self.image_proj_model(xy_embeds, c_embeds)
        c = ip_tokens.size(-1)
        ip_tokens = ip_tokens.reshape(b, -1, c)###########

        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, cross_attention_kwargs=cross_attention_kwargs,).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")

    
    
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default=None,
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )
    # parser.add_argument(
    #     "--data_json_file",
    #     type=str,
    #     default=None,
    #     required=True,
    #     help="Training data",
    # )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="",
        required=True,
        help="Training data root path",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        required=True,
        help="Path to CLIP image encoder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-ip_adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--noise_offset",
        type=float,
        default=0.1,
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
    

def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        gradient_accumulation_steps = 1,####1  约等于bs32了
    )
    
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    # image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    #image_encoder.requires_grad_(False)
    

    #ip-adapter
    image_proj_model = MLPProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        id_embeddings_dim=512,
        num_tokens=4  if is_multi else 4  # max length 
    )
    


    # init adapter modules
    lora_rank = 128
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        print(name)
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None: # attn1
            attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank)
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = LoRAIPAttnProcessor_p3(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank,num_tokens=4*3)#######单个的就是4
            attn_procs[name].load_state_dict(weights, strict=False)
    unet.set_attn_processor(attn_procs)#把atten换了
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())

    ###使用faceid-portrait的投影层
    ip_ckpt = "/root/code/IP-Adapter/models/ip-adapter-faceid_sd15.bin"#"/root/code/IP-Adapter/IP-Adapter/ip-adapter-faceid-portrait_sd15.bin"
    pretrained_model_state_dict = torch.load(ip_ckpt, map_location="cpu")["image_proj"]
    model_state_dict = image_proj_model.state_dict()

    # 更新模型的权重
    for name, weights in pretrained_model_state_dict.items():
        # print("&&&&&",name)
        if name in model_state_dict:
            print(name)
            model_state_dict[name].copy_(weights)

    # 加载权重到模型
    image_proj_model.load_state_dict(model_state_dict, strict=False)
    #pretrained_model_state_dict = torch.load(ip_ckpt, map_location="cpu")["ip_adapter"]
    #adapter_modules.load_state_dict(pretrained_model_state_dict, strict=True)

    unet = register_cross_attention_hook(unet)
    ip_adapter = IPAdapter(unet, image_proj_model, adapter_modules, args.pretrained_ip_adapter_path)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    #unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    #image_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # optimizer
    params_to_opt = itertools.chain(ip_adapter.image_proj_model.parameters(),  ip_adapter.adapter_modules.parameters())
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # dataloader
    train_dataset = MyDataset(tokenizer=tokenizer, size=args.resolution, image_root_path=args.data_root_path,split="train")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn_multi if is_multi else collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    loss_fn = BalancedL1Loss(threshold=1.0, normalize=False)#(args.object_localization_threshold,args.object_localization_normalize,)
    use_local_loss = False#True
    object_localization_weight = 0.001#
    show_step = 100
    loss_attn = None
    # Prepare everything with our `accelerator`.
    ip_adapter, optimizer, train_dataloader = accelerator.prepare(ip_adapter, optimizer, train_dataloader)
    
    global_step = 0
    for epoch in range(0, args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(ip_adapter):
                # Convert images to latent space
                with torch.no_grad():
                    latents = vae.encode(batch["images"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
                # image_embeds = batch["face_id_embed"].to(accelerator.device, dtype=weight_dtype)
                xy_embeds = batch["xy_embed"].to(accelerator.device, dtype=weight_dtype)
                c_embeds = batch["c_embed"].to(accelerator.device, dtype=weight_dtype)
                ip_attn_mask = batch["attn_mask"].to(accelerator.device, dtype=weight_dtype)
                ip_face_mask = batch["face_mask"].to(accelerator.device, dtype=weight_dtype)
                bbox_list = batch["bbox_list"]

                cross_attention_kwargs = {"face_masks":ip_face_mask, "bbox_list":bbox_list}
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["text_input_ids"].to(accelerator.device))[0]
                
                noise_pred = ip_adapter(noisy_latents, timesteps, encoder_hidden_states, xy_embeds, c_embeds,cross_attention_kwargs=cross_attention_kwargs,)
        
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
                if use_local_loss:
                    attn_maps = get_raw_attn_map()
                    loss_attn = 0
                    num_layers = len(attn_maps)
                    for name, attn_map in attn_maps.items():
                        #print(name, attn_map.shape)
                        layer_loss = get_object_localization_loss_for_one_layer(
                                attn_map, objmsks,loss_fn
                            )
                        loss_attn += layer_loss
                    loss_attn = loss_attn / num_layers
                    loss += object_localization_weight*loss_attn
                    clear_cross_attention_scores()

                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_main_process and step % show_step == 0:
                    print("Epoch {}, step {}, data_time: {}, time: {}, step_loss: {}, local_loss: {}".format(
                        epoch, step, load_data_time, time.perf_counter() - begin, avg_loss, loss_attn))
            
            global_step += 1
            
            if global_step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"Epoch-{epoch}-checkpoint-{global_step}")######
                # accelerator.save_state(save_path)
                
                # 保存参数
                state_dict = {"image_proj": {}, "ip_adapter": {}}
                state_dict["image_proj"] = ip_adapter.image_proj_model.state_dict()
                state_dict["ip_adapter"] = ip_adapter.adapter_modules.state_dict()
                os.makedirs(save_path, exist_ok=True)
                torch.save(state_dict, os.path.join(save_path, "ip_adapter.bin"))
            
            begin = time.perf_counter()
                
if __name__ == "__main__":
    main()    
