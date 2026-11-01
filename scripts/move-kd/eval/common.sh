#!/bin/bash
# Biến dùng chung cho scripts/move-kd/eval/*.sh - được 'source' từ các script benchmark.
#   $1    : đường dẫn checkpoint (mặc định ./checkpoints/llava-move-kd-7b-v1.1)
#   LIMIT : chỉ chạy N câu hỏi đầu (để test nhanh), bỏ trống = chạy toàn bộ
# Chạy từ thư mục gốc repo.
set -euo pipefail

CKPT="${1:-./checkpoints/llava-move-kd-7b-v1.1}"
CKPT="${CKPT%/}"
NAME="$(basename "$CKPT")"
LIMIT="${LIMIT:-}"
CONV_MODE="${CONV_MODE:-vicuna_v1}"

# builder.py chỉ nạp nhánh multimodal (EvalMoVELlavaLlamaForCausalLM + image processor)
# khi tên thư mục chứa 'llava' - xem download_ckpt.sh.
if [[ "${NAME,,}" != *llava* ]]; then
    echo "[FAIL] Tên checkpoint '$NAME' phải chứa 'llava' (llava/model/builder.py dựa vào tên để nạp vision tower)." >&2
    echo "       Hãy đổi tên thư mục, vd. checkpoints/llava-move-kd-7b-v1.1" >&2
    exit 1
fi

# Khi chạy tập con, gắn hậu tố vào tên file kết quả để không ghi đè kết quả full.
TAG="$NAME${LIMIT:+_limit$LIMIT}"

# subset_jsonl <file .jsonl> -> in ra đường dẫn file dùng để chạy (gốc hoặc N dòng đầu)
subset_jsonl() {
    local src="$1"
    if [[ -z "$LIMIT" ]]; then echo "$src"; return; fi
    local dst="${src%.jsonl}_limit${LIMIT}.jsonl"
    head -n "$LIMIT" "$src" > "$dst"
    echo "$dst"
}
