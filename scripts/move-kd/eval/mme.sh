#!/bin/bash
# MME (perception + cognition, 2374 câu yes/no). Dữ liệu: playground/data/eval/MME/{llava_mme.jsonl,MME_Benchmark_release_version/,convert_answer_to_mme.py,eval_tool/}
# Dùng: bash scripts/move-kd/eval/mme.sh [CKPT]
# Không hỗ trợ LIMIT: convert_answer_to_mme.py cần đủ câu trả lời cho mọi câu hỏi (bộ này chỉ ~20 phút trên 1 GPU).
LIMIT_REQUESTED="${LIMIT:-}"
unset LIMIT
source "$(dirname "$0")/common.sh"
[[ -n "$LIMIT_REQUESTED" ]] && echo "[WARN] mme.sh bỏ qua LIMIT=$LIMIT_REQUESTED, chạy toàn bộ MME."
DIR=./playground/data/eval/MME

python -m llava.eval.model_vqa_loader \
    --model-path "$CKPT" \
    --question-file $DIR/llava_mme.jsonl \
    --image-folder $DIR/MME_Benchmark_release_version \
    --answers-file $DIR/answers/$TAG.jsonl \
    --temperature 0 \
    --conv-mode $CONV_MODE

cd $DIR
python convert_answer_to_mme.py --experiment $TAG
cd eval_tool
python calculation.py --results_dir answers/$TAG | tee ../answers/${TAG}_score.txt
