#!/bin/bash
# POPE (ảo giác đối tượng, COCO val2014). Dữ liệu: playground/data/eval/pope/{llava_pope_test.jsonl,coco/,val2014/}
# Dùng: [LIMIT=50] bash scripts/move-kd/eval/pope.sh [CKPT]
source "$(dirname "$0")/common.sh"
DIR=./playground/data/eval/pope
QUESTIONS=$DIR/llava_pope_test.jsonl
if [[ -n "$LIMIT" ]]; then
    # eval_pope.py chấm cả 3 split và ghép đáp án với nhãn theo thứ tự (zip) -> lấy LIMIT câu ĐẦU của mỗi split,
    # không phải LIMIT dòng đầu file (toàn bộ thuộc một split, split còn lại chia cho 0)
    QUESTIONS=$DIR/llava_pope_test_limit$LIMIT.jsonl
    python - "$DIR/llava_pope_test.jsonl" "$QUESTIONS" "$LIMIT" <<'PY'
import collections, json, sys
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
seen = collections.Counter()
with open(dst, "w") as out:
    for line in open(src):
        cat = json.loads(line)["category"]
        if seen[cat] < n:
            seen[cat] += 1
            out.write(line)
PY
fi

python -m llava.eval.model_vqa_loader \
    --model-path "$CKPT" \
    --question-file "$QUESTIONS" \
    --image-folder $DIR/val2014 \
    --answers-file $DIR/answers/$TAG.jsonl \
    --temperature 0 \
    --conv-mode $CONV_MODE

python llava/eval/eval_pope.py \
    --annotation-dir $DIR/coco \
    --question-file "$QUESTIONS" \
    --result-file $DIR/answers/$TAG.jsonl | tee $DIR/answers/${TAG}_score.txt
