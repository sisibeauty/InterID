import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pdb import set_trace
attn_maps = [{} for _ in range(50)]
idxp = -1
count_name = 0
def set_idxp(idx):
    global idxp
    idxp = idx
def hook_fn(name):
    def forward_hook(module, input, output):
        global idxp
        global count_name
        if hasattr(module.processor, "attn_map"):
            count_name += 1
           # print(count_name,name)
            if name == "down_blocks.0.attentions.0.transformer_blocks.0.attn2":
                #print("here",idxp)
                idxp += 1
                # count_name += 1
                # if count_name == 16:
                #     idx += 1
                #     count_name = 0
            attn_maps[idxp][name] = module.processor.attn_map
            del module.processor.attn_map

    return forward_hook

def register_cross_attention_hook(unet):
    for name, module in unet.named_modules():
        if name.split('.')[-1].startswith('attn2'):
            module.register_forward_hook(hook_fn(name))

    return unet

def upscale(attn_map, target_size):
    attn_map = torch.mean(attn_map, dim=0)
    attn_map = attn_map.permute(1,0)
    temp_size = None

    for i in range(0,5):
        scale = 2 ** i
        if ( target_size[0] // scale ) * ( target_size[1] // scale) == attn_map.shape[1]*64:
            temp_size = (target_size[0]//(scale*8), target_size[1]//(scale*8))
            break

    assert temp_size is not None, "temp_size cannot is None"

    attn_map = attn_map.view(attn_map.shape[0], *temp_size)

    attn_map = F.interpolate(
        attn_map.unsqueeze(0).to(dtype=torch.float32),
        size=target_size,
        mode='bilinear',
        align_corners=False
    )[0]

    # attn_map = torch.softmax(attn_map, dim=0)
    return attn_map
def get_net_attn_map(image_size, batch_size=2, instance_or_negative=False, detach=True,start=0,end=30):

    idx = 0 if instance_or_negative else 1
    net_attn_maps = []
    # print(len(attn_maps),count_name,idx)
    # print(attn_maps[idx].keys())
    # print(attn_maps[0].keys())
    apmap = {}
    
    #set_trace()
    for i in range(start,end):
        for name,attn_map in attn_maps[i].items():
            if name not in apmap.keys():
                apmap[name] = attn_map
            else:
                apmap[name] += attn_map
    for name, attn_map in apmap.items():
        attn_map = attn_map / (end-start)
        attn_map = attn_map.cpu() if detach else attn_map
        attn_map = torch.chunk(attn_map, batch_size)[idx].squeeze()#
        #print("get",attn_map.shape)
        attn_map = upscale(attn_map, image_size) 
        #print(attn_map.shape)
        net_attn_maps.append(attn_map) 

    net_attn_maps = torch.mean(torch.stack(net_attn_maps,dim=0),dim=0)

    return net_attn_maps

def attnmaps2images(net_attn_maps):

    #total_attn_scores = 0
    images = []

    for attn_map in net_attn_maps:
        attn_map = attn_map.cpu().numpy()
        #total_attn_scores += attn_map.mean().item()

        normalized_attn_map = (attn_map - np.min(attn_map)) / (np.max(attn_map) - np.min(attn_map)) * 255
        normalized_attn_map = normalized_attn_map.astype(np.uint8)
        #print("norm: ", normalized_attn_map.shape)
        image = Image.fromarray(normalized_attn_map)

        #image = fix_save_attn_map(attn_map)
        images.append(image)

    #print(total_attn_scores)
    return images
def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")

def get_generator(seed, device):

    if seed is not None:
        if isinstance(seed, list):
            generator = [torch.Generator(device).manual_seed(seed_item) for seed_item in seed]
        else:
            generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = None

    return generator