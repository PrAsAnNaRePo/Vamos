git lfs install
git clone https://huggingface.co/NousResearch/Nous-Capybara-3B-V1.9

mkdir base_ckpt
mv Nous-Capybara-3B-V1.9/model-00001-of-00003.safetensors base_ckpt/
mv Nous-Capybara-3B-V1.9/model-00002-of-00003.safetensors base_ckpt/
mv Nous-Capybara-3B-V1.9/model-00003-of-00003.safetensors base_ckpt/
mv Nous-Capybara-3B-V1.9/config.json base_ckpt/
mv Nous-Capybara-3B-V1.9/configuration_stablelm_epoch.py base_ckpt/
mv Nous-Capybara-3B-V1.9/modeling_stablelm_epoch.py base_ckpt/
mv Nous-Capybara-3B-V1.9/model.safetensors.index.json base_ckpt/

rm -r Nous-Capybara-3B-V1.9

torchrun --nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6001 pretrain/hf_train.py \
    --bf16 True \
    --max_len 55 \
    --eval_steps 5000 \
    --dataset_id "coco" \
    --output_dir "vamos" \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 8 \
    --evaluation_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 3 \
    --learning_rate 1e-4 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --deepspeed pretrain/zero2.json