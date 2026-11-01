#!/bin/bash
# Eval baseline LLaVA-1.5 bằng code LLaVA gốc, dùng chung dữ liệu và các wrapper eval của repo này.
# Không eval được checkpoint LLaVA-1.5 bằng code MoVE-KD (builder.py nạp nó thành EvalMoVELlavaLlamaForCausalLM,
# gắn LoRA expert ngẫu nhiên vào CLIP - xem Experiments.md mục 2.6). Đặt LLaVA gốc lên PYTHONPATH cũng không đủ:
# 'python -m' luôn ưu tiên thư mục hiện tại, nên vẫn nạp llava/ của repo này. Vì vậy script chạy các wrapper
# từ thư mục LLaVA gốc, với playground/ và scripts/move-kd/ là symlink về repo này.
#
# Dùng (ở thư mục gốc repo): bash scripts/move-kd/eval/baseline.sh [CKPT] [benchmark ...]
#   CKPT mặc định liuhaotian/llava-v1.5-7b; benchmark mặc định: pope mme textvqa sqa
#   LLAVA_DIR=<nơi clone LLaVA gốc> (mặc định ../LLaVA, tự clone nếu chưa có)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
LLAVA_DIR="${LLAVA_DIR:-$REPO/../LLaVA}"
CKPT="${1:-liuhaotian/llava-v1.5-7b}"
shift || true
BENCHES=("$@")
[[ ${#BENCHES[@]} -eq 0 ]] && BENCHES=(pope mme textvqa sqa)
# đường dẫn local tương đối -> tuyệt đối, vì các wrapper sẽ chạy từ thư mục LLaVA gốc
[[ -e "$CKPT" ]] && CKPT="$(cd "$CKPT" && pwd)"

[[ -d "$LLAVA_DIR" ]] || git clone -q https://github.com/haotian-liu/LLaVA "$LLAVA_DIR"
cd "$LLAVA_DIR"
ln -sfn "$REPO/playground" playground
mkdir -p scripts && ln -sfn "$REPO/scripts/move-kd" scripts/move-kd
python -c "import llava, os, sys; f = os.path.realpath(llava.__file__); print('[ OK ] llava từ', f); \
sys.exit(0 if f.startswith(os.path.realpath('.') + os.sep) else 'llava không phải bản LLaVA gốc')"

for b in "${BENCHES[@]}"; do
    bash scripts/move-kd/eval/$b.sh "$CKPT"
done
