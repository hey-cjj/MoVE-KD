#!/bin/bash
# TextVQA val (OCR). Dữ liệu: playground/data/eval/textvqa/{llava_textvqa_val_v051_ocr.jsonl,TextVQA_0.5.1_val.json,train_images/}
# Dùng: [LIMIT=50] bash scripts/move-kd/eval/textvqa.sh [CKPT]
source "$(dirname "$0")/common.sh"
DIR=./playground/data/eval/textvqa
QUESTIONS="$(subset_jsonl $DIR/llava_textvqa_val_v051_ocr.jsonl)"

python -m llava.eval.model_vqa_loader \
    --model-path "$CKPT" \
    --question-file "$QUESTIONS" \
    --image-folder $DIR/train_images \
    --answers-file $DIR/answers/$TAG.jsonl \
    --temperature 0 \
    --conv-mode $CONV_MODE

python -m llava.eval.eval_textvqa \
    --annotation-file $DIR/TextVQA_0.5.1_val.json \
    --result-file $DIR/answers/$TAG.jsonl | tee $DIR/answers/${TAG}_score.txt
