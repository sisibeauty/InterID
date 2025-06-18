#rm -rf "/root/code/IP-Adapter/multiface_pre/logs/2.29_preexpri_1.1_noise"
accelerate launch --num_processes 4 --gpu_ids='0,1,2,3' --mixed_precision "fp16" \
  /root/code/InterID-code/train_faceid_multi_pose.py \
  --pretrained_model_name_or_path="/root/base_model/stable-diffusion-v1-5/" \
  --image_encoder_path="models/image_encoder/" \
  --data_root_path="/data/data/merged" \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=8 \
  --dataloader_num_workers=1 \
  --learning_rate=1e-04 \
  --weight_decay=0.01 \
  --output_dir="/root/code/IP-Adapter/multiface/logs/810_12_4token_local__pose_faceidnocros_nodrop_5_1.0_noise" \
  --save_steps=4000 \
  --noise_offset 0 \
  --num_train_epochs=250

