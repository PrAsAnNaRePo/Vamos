torchrun --nproc_per_node 4 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6001 pretrain/hf_train.py \
    --fp16 True \
    --llm_id "HuggingFaceH4/zephyr-7b-beta" \
    --max_len 55 \
    --eval_steps 1500 \
    --dataset_id "textcap" \
    --output_dir "vamos" \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 8 \
    --evaluation_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 3 \
    --learning_rate 5e-3 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "wandb" \
    --deepspeed pretrain/zero2.json