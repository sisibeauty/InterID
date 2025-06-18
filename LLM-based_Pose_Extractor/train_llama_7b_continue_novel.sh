base_model=/root/base_model/llama-7B/
#model=chinese-llama-13b-continue-novel-persona-js-1106
#corpus_dir=/project/corpus/persona_js_1106/
#model_dir=/project/checkpoints/
model=llama-7B-pose-6.7
corpus_dir=/data/data/llama_data/
model_dir=/data/llm-ckpts/llm-6-7/
tensorboard_dir=/root/logs/llama-6.7/
logs_dir=/root/logs/
torchrun --nproc_per_node 2 --master_port=9901 \
        examples/pytorch/language-modeling/run_clm.py \
        --model_name_or_path ${base_model} \
        --train_file ${corpus_dir}/train.json \
        --validation_file ${corpus_dir}/test.json \
        --fp16 --per_device_train_batch_size 2 --gradient_accumulation_steps 1 \
        --warmup_steps 200 --use_fast_tokenizer False \
        --learning_rate 1e-05 --block_size 2048 \
        --do_train --gradient_checkpointing --num_train_epochs 150 --do_eval --resume_from_checkpoint=/data/llm-ckpts/llm-6-7/llama-7B-pose-6.7/checkpoint-78000\
        --output_dir ${model_dir}/${model} --overwrite_output_dir \
        --logging_dir ${tensorboard_dir}/${model}/ --logging_steps 50 \
        --deepspeed ds_config_zero3.json | tee -a ${logs_dir}/${model}.txt
#--resume_from_checkpoint=/root/checkpoints/llama-7B-pose/checkpoint-31000\#you can fine-tune starting from raw llama-7B