#!/bin/bash
# ScienceQA-IMG (test). Dữ liệu: playground/data/eval/scienceqa/{llava_test_CQM-A.json,pid_splits.json,problems.json,images/test/}
# Dùng: [LIMIT=50] bash scripts/move-kd/eval/sqa.sh [CKPT]
source "$(dirname "$0")/common.sh"
DIR=./playground/data/eval/scienceqa
QUESTIONS=$DIR/llava_test_CQM-A.json
if [[ -n "$LIMIT" ]]; then
    python -c "import json,sys; json.dump(json.load(open(sys.argv[1]))[:int(sys.argv[3])], open(sys.argv[2],'w'))" \
        $QUESTIONS $DIR/llava_test_CQM-A_limit$LIMIT.json $LIMIT
    QUESTIONS=$DIR/llava_test_CQM-A_limit$LIMIT.json
fi

python -m llava.eval.model_vqa_science \
    --model-path "$CKPT" \
    --question-file $QUESTIONS \
    --image-folder $DIR/images/test \
    --answers-file $DIR/answers/$TAG.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode $CONV_MODE

python llava/eval/eval_science_qa.py \
    --base-dir $DIR \
    --result-file $DIR/answers/$TAG.jsonl \
    --output-file $DIR/answers/${TAG}_output.jsonl \
    --output-result $DIR/answers/${TAG}_result.json | tee $DIR/answers/${TAG}_score.txt

# Paper báo cáo SQA-IMG -> đọc "IMG-Accuracy". Chỉ số này vẫn đúng khi chạy LIMIT vì câu chưa chạy
# có prompt 'Unknown' nên không được tính là câu có ảnh (còn "Accuracy" tổng thì bị kéo xuống).
