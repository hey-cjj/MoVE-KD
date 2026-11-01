#!/bin/bash
# Smoke-test pipeline Stage 1 (pretrain-v1.1) trên 1 GPU: 4 teacher (CLIP, SAM, EVA-02, ConvNeXt) + MoLE + encoder adapters.
# Thay Vicuna-7B bằng TinyLlama-1.1B và chỉ chạy vài chục bước trên N mẫu LCS-558K.
# Mục đích: chứng minh code train chạy được (forward KD, backward, lưu mole_encoder.bin/encoder_adapter.bin),
# KHÔNG phải để tái lập số liệu paper.
#
# Yêu cầu:
#  * GPU hỗ trợ bf16 (Ampere+: A100/A10/L4/RTX 30xx-40xx), ~24GB. initialize_teachers() hardcode teacher
#    và adapter ở bf16, nên chạy fp16 (T4/V100) sẽ lệch dtype.
#  * Đã chạy: DOWNLOAD_CKPT=1 DOWNLOAD_DATA=1 bash setup.sh (teacher ở ./checkpoints, dữ liệu ở playground/data)
#
# Dùng (ở thư mục gốc repo): [N_SAMPLES=200 MAX_STEPS=30] bash scripts/move-kd/smoke_pretrain.sh
set -euo pipefail

N_SAMPLES="${N_SAMPLES:-200}"
MAX_STEPS="${MAX_STEPS:-30}"
LLM="${LLM:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
OUT=./checkpoints/smoke/move-kd-tinyllama-v1.1
DATA=./playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k_${N_SAMPLES}.json
mkdir -p "$(dirname "$OUT")"

python -c "import json,sys; json.dump(json.load(open(sys.argv[1]))[:int(sys.argv[3])], open(sys.argv[2],'w'))" \
    ./playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json "$DATA" "$N_SAMPLES"

# Có flash-attn (sm80+) -> train_mem.py như script gốc; không có -> train.py (attention mặc định)
TRAIN=llava/train/train_mem.py
python -c "import flash_attn" 2>/dev/null || TRAIN=llava/train/train.py

deepspeed --num_gpus 1 $TRAIN \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path "$LLM" \
    --version plain \
    --data_path "$DATA" \
    --image_folder ./playground/data/LLaVA-Pretrain/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $OUT \
    --max_steps $MAX_STEPS \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --tune_encoder True \
    --tune_encoder_adapter True \
    --encoder_teachers clip sam eva convnext \
    --moe_encoder True \
    --moe_encoder_dense False \
    --moe_encoder_lora_rank 32 \
    --moe_encoder_lora_alpha 1 \
    --moe_encoder_num_experts 3 \
    --moe_encoder_balance_w 0.01 \
    --kd True \
    --kd_w 0.5 \
    --kd_memory_w 0.8 \
    --token_weight True \
    --teacher_weight True \
    --report_to none 2>&1 | tee $OUT.log

echo
for f in mm_projector.bin mole_encoder.bin encoder_adapter.bin; do
    [[ -f $OUT/$f ]] && echo "[ OK ] $OUT/$f" || echo "[MISS] $OUT/$f"
done
echo "Loss (tổng = LM + 0.01*balance + 0.5*KD) theo bước:"
grep -o "'loss': [0-9.]*" $OUT.log | awk '{print NR": "$2}' | sed -n '1p;$p'
