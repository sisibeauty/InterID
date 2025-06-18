import torch
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
            image_ids_path = os.path.join(image_root_path, "image_ids_train.txt")
        elif split == "test":
            image_ids_path = os.path.join(image_root_path, "image_ids_test.txt")
        else:
            raise ValueError(f"Unknown split {split}")

        with open(image_ids_path, "r") as f:
            self.image_ids = f.read().splitlines()
            
        print("data dir:", image_root_path)
        print("file num:", len(self.image_ids))
        # for file in tqdm(files):
        #     if file.endswith(".json"):
        #         with open(os.path.join(image_root_path, file), "r") as f:
        #             xyc_list = json.load(f)
        #         if not is_multi:
        #             self.data.append({
        #             "text": "A circle on a solid black background",
        #             "image_file": file.replace(".json", ".png"),
        #             "xyc_list": xyc_list
        #             })
        #         else:
        #             # tt = "A circle"
        #             # for poin in range(len(xyc_list)-1):
        #             #     tt += "and a circle"

        #             self.data.append({
        #             "text": "A circle on a solid black background",
        #             "image_file": file.replace(".json", ".png"),
        #             "xyc_list": xyc_list
        #             })


        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        chunk = image_id[:5]
        image_path = os.path.join(self.image_root_path, chunk, image_id, image_id + ".jpg")
        info_path = os.path.join(self.image_root_path, chunk, image_id, image_id + ".json")

        with open(info_path, "r") as f:
            info_dict = json.load(f)#caption,xy

       
        text = info_dict['caption']
    
        
        # read image
        raw_image = Image.open(image_path)
        image = self.transform(raw_image.convert("RGB"))

        xy_embeds = []
        face_embeds = []
        info_dict = info_dict["pose"][0]
        face_num = len(info_dict)# - 3#1,2
        face_num = min(face_num,5)##数据集最多有6张
        for idx in range(face_num):
            xy_embed = torch.tensor(info_dict[str(idx)], dtype=torch.float32) / 511.0
            xy_embed = xy_embed.reshape(xy_embed.shape[0]*xy_embed.shape[1])
            #c_embed = torch.tensor(c, dtype=torch.float32) / 255.0
            face_path = os.path.join(self.image_root_path, chunk, image_id, str(idx) + ".npy")
            face_id_embed = np.load(face_path,allow_pickle=True)
            face_id_embed = torch.from_numpy(face_id_embed)
            # drop
            drop_image_embed = 0
            rand_num = random.random()
            if rand_num < self.i_drop_rate:
                drop_image_embed = 1
            elif rand_num < (self.i_drop_rate + self.t_drop_rate):
                text = ""
            elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
                text = ""
                drop_image_embed = 1
            if drop_image_embed:
                # xy_embed = torch.zeros_like(xy_embed)
                #c_embed = torch.full_like(c_embed, -1)
                face_id_embed = torch.zeros_like(face_id_embed)

            rand_num = random.random()############################随机drop xy
            if rand_num < self.xy_drop_rate:
                xy_embed = torch.zeros_like(xy_embed)#torch.full_like(xy_embed, -1)#

            # get text and tokenize
            text_input_ids = self.tokenizer(
                text,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids
            
            xy_embeds.append(xy_embed)
            face_embeds.append(face_id_embed)#######


        while len(face_embeds) < 5:
            #print(len(face_embeds))
            try:
                face_embeds.append(torch.full_like(face_embeds[0], 0))
                xy_embeds.append(torch.full_like(xy_embeds[0], -1))#########padding -1
            except:
                print(image_id,info_dict,face_num)
           
        
            # print("*************157****")
            # print(len(c_embeds),len(c_embeds[0]),len(xy_embeds))#5 3 5
        face_embeds = torch.stack(face_embeds)
        xy_embeds = torch.stack(xy_embeds)
        #print(info_path)
        #print(xy_embeds)
        #print(face_embeds)
        #print("*************161****")
        #print(face_embeds.shape,xy_embeds.shape)#torch.Size([5, 512]) torch.Size([5, 4])
        #x = input()

        # print("*************173****")
        # print(c_embeds.shape,xy_embeds.shape)#torch.Size([5, 3]) torch.Size([5, 2])
        return {
                "image": image,
                "text_input_ids": text_input_ids,
                "drop_image_embed": drop_image_embed,
                "xy_embed": xy_embeds,
                "c_embed": face_embeds
            }


    def __len__(self):
        return len(self.image_ids)
    

def collate_fn_multi(data):
    images = torch.stack([example["image"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    xy_embed = torch.stack([example["xy_embed"] for example in data])
    c_embed = torch.stack([example["c_embed"] for example in data])
    drop_image_embeds = [example["drop_image_embed"] for example in data]
    # print("*****179******",xy_embed.shape)
    # print(c_embed.shape)#torch.Size([8, 5, 512])
    return {
        "images": images,
        "text_input_ids": text_input_ids,
        "xy_embed": xy_embed,
        "c_embed": c_embed,
        "drop_image_embeds": drop_image_embeds
    }
    

def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    xy_embed = torch.stack([example["xy_embed"] for example in data])
    c_embed = torch.stack([example["c_embed"] for example in data])
    drop_image_embeds = [example["drop_image_embed"] for example in data]
    #print("*****196******",xy_embed.shape)
    # print(c_embed.shape)#torch.Size([8, 2]) torch.Size([8, 3])

    return {
        "images": images,
        "text_input_ids": text_input_ids,
        "xy_embed": xy_embed,
        "c_embed": c_embed,
        "drop_image_embeds": drop_image_embeds
    }