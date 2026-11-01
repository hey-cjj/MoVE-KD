#!/bin/bash
# Tải checkpoint MoVE-KD đã công bố về thư mục local có tên chứa 'llava'.
# llava/model/builder.py chỉ nạp nhánh multimodal (EvalMoVELlavaLlamaForCausalLM + image processor)
# khi tên thư mục cuối chứa 'llava'; dùng thẳng 'jiaojuncao/MoVE-KD-7b-v1.1' sẽ nạp nhầm thành LLM thuần.
#
# Dùng: bash scripts/move-kd/eval/download_ckpt.sh [repo_id ...]   (mặc định: jiaojuncao/MoVE-KD-7b-v1.1)
#
# Chỉ dành cho checkpoint MoVE-KD. KHÔNG dùng repo này để eval checkpoint LLaVA-1.5 gốc: builder.py nạp mọi
# model tên 'llava' bằng EvalMoVELlavaLlamaForCausalLM, config LLaVA-1.5 không có khoá 'moe' nên mặc định
# moe_encoder=True -> CLIP bị gắn LoRA expert khởi tạo ngẫu nhiên; còn kd_mode=False thì encode_images lỗi.
# Baseline: bash scripts/move-kd/eval/baseline.sh liuhaotian/llava-v1.5-7b (chạy wrapper bằng code LLaVA gốc)
set -euo pipefail
cd "$(dirname "$0")/../../.."

REPOS=("$@")
[[ ${#REPOS[@]} -eq 0 ]] && REPOS=(jiaojuncao/MoVE-KD-7b-v1.1)

for repo in "${REPOS[@]}"; do
    name="$(basename "$repo" | tr '[:upper:]' '[:lower:]')"
    [[ "$name" == *llava* ]] || name="llava-$name"
    python - "$repo" "checkpoints/$name" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, out = sys.argv[1], sys.argv[2]
# training_args.bin / trainer_state.json không cần cho inference
snapshot_download(repo, local_dir=out, ignore_patterns=["training_args.bin", "trainer_state.json", "*.png"])
print(f"[ OK ] {repo} -> {out}")
PY
done
