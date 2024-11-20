# lora
deepspeed --include localhost:1,2,3,4 --master_port 520 ./train_sft.py \
                --data_path data/data.json \
                --per_device_train_batch_size 1 \
                --max_len 128000 \
                --is_skip \
                --learning_rate 1e-4 \
                --weight_decay 0.1 \
                --num_train_epochs 2 \
                --gradient_accumulation_steps 4 \
                --warmup_ratio 0.1 \
                --train_type lora \
                --lora_dim 16 \
                --lora_alpha 64 \
                --lora_dropout 0.1 \
                --lora_module_name "query_key_value,dense_h_to_4h,dense_4h_to_h,dense" \
                --seed 1234 \
                --ds_file ds_config/ds_zero2_no_offload.json \
                --gradient_checkpointing \
                --show_loss_step 10 \
                --save_model_step 200 \
                --output_dir out

# 全量sft
# deepspeed --include localhost:1,2,3,4 --master_port 520 ./train_sft.py \
#               --train_path data/data.json \
#               --per_device_train_batch_size 1 \
#               --max_len 128000 \
#               --learning_rate 1e-4 \
#               --weight_decay 0.1 \
#               --num_train_epochs 2 \
#               --gradient_accumulation_steps 4 \
#               --warmup_ratio 0.1 \
#               --train_type all \
#               --seed 1234 \
#               --ds_file ds_config/ds_zero2_no_offload.json \
#               --show_loss_step 10 \
#               --save_model_step 200 \
#               --output_dir out
