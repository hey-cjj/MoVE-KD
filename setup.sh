#!/usr/bin/env bash
# =============================================================================
# setup.sh - Thiết lập môi trường huấn luyện MoVE-KD
#            (PyTorch 2.1.2 + DeepSpeed ZeRO-2 + Flash-Attention 2)
#
# Chạy trên Linux + GPU NVIDIA (khuyến nghị Ampere trở lên: A100/H100/A6000/
# RTX 30xx/40xx để dùng Flash-Attention 2 và bf16).
#
# Cách dùng (tại thư mục gốc repo):
#   bash setup.sh                    # tạo env + cài thư viện + sinh config DeepSpeed + kiểm tra
#   DOWNLOAD_CKPT=1 bash setup.sh    # + tải checkpoint teacher (SAM, EVA-02, ConvNeXt), CLIP, Vicuna
#   DOWNLOAD_DATA=1 bash setup.sh    # + tải dữ liệu pre-train LCS-558K và file json fine-tune
#   SKIP_CONDA=1 bash setup.sh       # cài vào Python hiện tại (Docker/Colab), không tạo conda env
#
# Biến môi trường tuỳ chỉnh:
#   ENV_NAME=move-kd  PYTHON_VERSION=3.10  CUDA_TAG=cu121 (hoặc cu118)
#   FLASH_ATTN_VERSION=2.5.8  MAX_JOBS=4 (số job khi phải build flash-attn từ source)
#   FORCE_FLASH_ATTN=1 (cài flash-attn kể cả khi GPU < sm80)
#   HF_HOME=... / HF_ENDPOINT=... (cache & mirror của Hugging Face)
# =============================================================================
set -euo pipefail

ENV_NAME="${ENV_NAME:-move-kd}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CUDA_TAG="${CUDA_TAG:-cu121}"
TORCH_VERSION="2.1.2"
TORCHVISION_VERSION="0.16.2"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.5.8}"
DOWNLOAD_CKPT="${DOWNLOAD_CKPT:-0}"
DOWNLOAD_DATA="${DOWNLOAD_DATA:-0}"
SKIP_CONDA="${SKIP_CONDA:-0}"
FORCE_FLASH_ATTN="${FORCE_FLASH_ATTN:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

info()  { printf '\033[1;34m[INFO]\033[0m %s\n' "$*"; }
ok()    { printf '\033[1;32m[ OK ]\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m %s\n' "$*"; }
die()   { printf '\033[1;31m[FAIL]\033[0m %s\n' "$*" >&2; exit 1; }
step()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# -----------------------------------------------------------------------------
step "1/7 Kiểm tra hệ thống"
# -----------------------------------------------------------------------------
[[ "$(uname -s)" == "Linux" ]] || die "MoVE-KD (DeepSpeed + Flash-Attention) chỉ hỗ trợ Linux. Trên Windows hãy dùng WSL2 hoặc server Linux."
[[ -f pyproject.toml && -d llava ]] || die "Hãy đặt setup.sh ở thư mục gốc repo MoVE-KD."

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader \
        | while IFS= read -r line; do info "GPU: $line"; done
else
    warn "Không tìm thấy nvidia-smi - máy này không có GPU/driver NVIDIA. Vẫn cài được thư viện nhưng không train được."
fi

if command -v nvcc >/dev/null 2>&1; then
    info "nvcc: $(nvcc --version | grep -o 'release [0-9.]*')"
else
    warn "Không có nvcc (CUDA toolkit). Không sao nếu flash-attn có wheel dựng sẵn; nếu phải build từ source thì cần CUDA toolkit >= 11.8."
fi

# -----------------------------------------------------------------------------
step "2/7 Tạo/kích hoạt môi trường Python"
# -----------------------------------------------------------------------------
if [[ "$SKIP_CONDA" == "1" ]]; then
    info "SKIP_CONDA=1 -> dùng Python hiện tại: $(command -v python)"
else
    command -v conda >/dev/null 2>&1 || die "Chưa có conda. Cài Miniconda: https://docs.anaconda.com/miniconda/ (hoặc chạy lại với SKIP_CONDA=1)."
    # Script activate của conda tham chiếu biến chưa khai báo -> tạm tắt 'set -u'
    set +u
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if conda env list | awk '{print $1}' | grep -x "$ENV_NAME" >/dev/null; then
        info "Env '$ENV_NAME' đã tồn tại -> dùng lại."
    else
        info "Tạo env '$ENV_NAME' (python=$PYTHON_VERSION)..."
        conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
    fi
    conda activate "$ENV_NAME"
    set -u
fi
PY_VER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[[ "$PY_VER" == "3.10" || "$PY_VER" == "3.11" || "$PY_VER" == "3.9" ]] \
    || warn "Python $PY_VER chưa được kiểm thử với torch $TORCH_VERSION (khuyến nghị 3.10)."
ok "Python $PY_VER tại $(command -v python)"

# setuptools>=81 bỏ pkg_resources -> setup.py của flash-attn 2.5.8 (build --no-build-isolation) lỗi
python -m pip install --upgrade pip wheel "setuptools<70" packaging

# -----------------------------------------------------------------------------
step "3/7 Cài PyTorch $TORCH_VERSION ($CUDA_TAG)"
# -----------------------------------------------------------------------------
# Cài torch trước, từ index của PyTorch, để chắc chắn đúng bản CUDA.
# pyproject.toml ghim torch==2.1.2 nên bước 'pip install -e' phía sau sẽ giữ nguyên bản này.
python -m pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

# -----------------------------------------------------------------------------
step "4/7 Cài MoVE-KD (llava) + DeepSpeed + thư viện phụ"
# -----------------------------------------------------------------------------
# [train] = deepspeed==0.12.6, ninja, wandb
python -m pip install -e ".[train]"

# Các bản ghim bổ sung - pyproject.toml gốc (kế thừa từ LLaVA) chưa đủ cho MoVE-KD:
#  * timm==0.9.16 : teacher ConvNeXt (vision_models/convnext.py) import 'timm.layers' và
#                   'timm.models._registry' -> cần timm >= 0.9 (pyproject ghim 0.6.13).
#                   pip sẽ báo "llava requires timm==0.6.13" - cảnh báo này có thể bỏ qua.
#  * fvcore, scipy : teacher EVA-02 (vision_models/eva_vit.py).
#  * tensorboard   : các script move-kd dùng --report_to tensorboard.
#  * numpy<2       : torch 2.1.2 và scikit-learn 1.2.2 được build với NumPy 1.x.
#  * peft==0.10.0  : peft mới hơn đòi transformers mới hơn 4.37.2.
python -m pip install \
    "timm==0.9.16" \
    "numpy==1.26.4" \
    "peft==0.10.0" \
    fvcore scipy tensorboard

# -----------------------------------------------------------------------------
step "5/7 Cài Flash-Attention $FLASH_ATTN_VERSION"
# -----------------------------------------------------------------------------
# train_mem.py gọi train(attn_implementation="flash_attention_2") -> bắt buộc flash-attn 2.x
# và GPU kiến trúc Ampere (sm80) trở lên.
GPU_CC="$(python - <<'EOF' 2>/dev/null || true
import torch
if torch.cuda.is_available():
    print(min(torch.cuda.get_device_capability(i)[0] for i in range(torch.cuda.device_count())))
EOF
)"
if [[ -n "$GPU_CC" && "$GPU_CC" -lt 8 && "$FORCE_FLASH_ATTN" != "1" ]]; then
    warn "GPU có compute capability sm${GPU_CC}x < sm80 -> Flash-Attention 2 không chạy được, bỏ qua bước này."
    warn "Khi train, thay 'llava/train/train_mem.py' bằng 'llava/train/train.py' trong scripts/move-kd/*.sh"
    warn "và đổi '--bf16 True --tf32 True' thành '--fp16 True --tf32 False' (V100/T4 không hỗ trợ bf16/tf32)."
else
    [[ -z "$GPU_CC" ]] && info "Không phát hiện được GPU (vd. login node) -> vẫn cài flash-attn."
    MAX_JOBS="${MAX_JOBS:-4}" python -m pip install "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation
fi

# -----------------------------------------------------------------------------
step "6/7 Sinh file cấu hình DeepSpeed ZeRO-2"
# -----------------------------------------------------------------------------
# scripts/move-kd/*.sh tham chiếu ./scripts/zero2.json và ./scripts/zero2_finetune.json,
# nhưng .gitignore của repo bỏ qua mọi *.json nên 2 file này không có sẵn sau khi clone.
# Cấu hình dưới đây là ZeRO-2 chuẩn của LLaVA; các giá trị "auto" lấy từ tham số HF Trainer.
write_zero2() {
    local path="$1"
    if [[ -f "$path" ]]; then
        info "$path đã tồn tại -> giữ nguyên."
        return
    fi
    cat > "$path" <<'EOF'
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "bf16": {
        "enabled": "auto"
    },
    "train_micro_batch_size_per_gpu": "auto",
    "train_batch_size": "auto",
    "gradient_accumulation_steps": "auto",
    "zero_optimization": {
        "stage": 2,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto"
    }
}
EOF
    ok "Đã tạo $path"
}
write_zero2 scripts/zero2.json
write_zero2 scripts/zero2_finetune.json

# -----------------------------------------------------------------------------
step "7/7 (Tuỳ chọn) Tải checkpoint & dữ liệu"
# -----------------------------------------------------------------------------
# Đường dẫn teacher được hardcode trong llava/model/multimodal_encoder/mixture_vision_models.py:
#   SAM      -> ./checkpoints/sam/sam_vit_l                        (định dạng HF: facebook/sam-vit-large)
#   EVA-02   -> ./checkpoints/EVA/eva02/det/eva02_L_coco_det_sys_o365.pth
#   ConvNeXt -> timm tự tải 'convnext_xxlarge' từ HF Hub vào cache
#   CLIP     -> openai/clip-vit-large-patch14-336 (vừa là student khởi tạo, vừa là teacher)
if [[ "$DOWNLOAD_CKPT" == "1" ]]; then
    python - <<'EOF'
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

def fetch(repo_id, local_dir=None):
    # Ưu tiên safetensors để không tải trùng trọng số dạng .bin
    ignore = ["*.h5", "*.msgpack", "*.onnx", "*.ot", "*.tflite"]
    if any(f.endswith(".safetensors") for f in list_repo_files(repo_id)):
        ignore.append("*.bin")
    path = snapshot_download(repo_id, local_dir=local_dir, ignore_patterns=ignore)
    print(f"[ OK ] {repo_id} -> {path}")

fetch("facebook/sam-vit-large", local_dir="checkpoints/sam/sam_vit_l")
p = hf_hub_download("Yuxin-CV/EVA-02", "eva02/det/eva02_L_coco_det_sys_o365.pth",
                    local_dir="checkpoints/EVA")
print(f"[ OK ] EVA-02-L det -> {p}")
fetch("openai/clip-vit-large-patch14-336")
fetch("lmsys/vicuna-7b-v1.5")
try:
    # Tag mặc định mà timm dùng cho 'convnext_xxlarge' khi pretrained=True
    fetch("timm/convnext_xxlarge.clip_laion2b_soup_ft_in1k")
except Exception as e:
    print(f"[WARN] Không prefetch được ConvNeXt ({e}); timm sẽ tự tải ở lần train đầu tiên.")
EOF
else
    info "Bỏ qua tải checkpoint (đặt DOWNLOAD_CKPT=1 để tải)."
fi

if [[ "$DOWNLOAD_DATA" == "1" ]]; then
    command -v unzip >/dev/null 2>&1 || die "Cần 'unzip' để giải nén dữ liệu (sudo apt install unzip)."
    python - <<'EOF'
from huggingface_hub import hf_hub_download
for repo, fname, out in [
    ("liuhaotian/LLaVA-Pretrain", "blip_laion_cc_sbu_558k.json", "playground/data/LLaVA-Pretrain"),
    ("liuhaotian/LLaVA-Pretrain", "images.zip", "playground/data/LLaVA-Pretrain"),
    ("liuhaotian/LLaVA-Instruct-150K", "llava_v1_5_mix665k.json", "playground/data"),
]:
    print("[ OK ]", hf_hub_download(repo, fname, repo_type="dataset", local_dir=out))
EOF
    if [[ ! -d playground/data/LLaVA-Pretrain/images ]]; then
        unzip -q playground/data/LLaVA-Pretrain/images.zip -d playground/data/LLaVA-Pretrain/images
    fi
    ok "Dữ liệu pre-train sẵn sàng tại playground/data/LLaVA-Pretrain"
    warn "Ảnh fine-tune (COCO, GQA, OCR-VQA, TextVQA, VisualGenome) cần tải thủ công vào playground/data/,"
    warn "xem: https://github.com/haotian-liu/LLaVA#visual-instruction-tuning"
else
    info "Bỏ qua tải dữ liệu (đặt DOWNLOAD_DATA=1 để tải)."
fi

# -----------------------------------------------------------------------------
step "Kiểm tra môi trường"
# -----------------------------------------------------------------------------
python - <<'EOF'
import importlib, os

def ver(mod):
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "ok")
    except Exception as e:
        return f"LỖI: {type(e).__name__}: {e}"

import torch
print(f"torch          {torch.__version__} (CUDA {torch.version.cuda})")
print(f"CUDA available {torch.cuda.is_available()}  | #GPU = {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"bf16 support   {torch.cuda.is_bf16_supported()}")
for mod in ["transformers", "tokenizers", "accelerate", "deepspeed", "flash_attn",
            "peft", "timm", "fvcore", "numpy", "tensorboard"]:
    print(f"{mod:<14} {ver(mod)}")

# Import chính mô hình MoVE-KD: kiểm tra toàn bộ chuỗi import (MoLE, teacher encoders, timm.layers...)
try:
    from llava.model.language_model.move_llava_llama import MoVELlavaLlamaForCausalLM  # noqa: F401
    from llava.model.multimodal_encoder.vision_models.convnext_encoder import ConvNextVisionTower  # noqa: F401
    from llava.model.multimodal_encoder.vision_models.eva_vit import EVAVITVisionTower  # noqa: F401
    print("[ OK ] import MoVELlavaLlamaForCausalLM + teacher encoders")
except Exception as e:
    print(f"[FAIL] import mô hình MoVE-KD: {type(e).__name__}: {e}")

for path in ["scripts/zero2.json", "scripts/zero2_finetune.json",
             "checkpoints/sam/sam_vit_l/config.json",
             "checkpoints/EVA/eva02/det/eva02_L_coco_det_sys_o365.pth",
             "playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json"]:
    print(f"[{' OK ' if os.path.exists(path) else 'MISS'}] {path}")
EOF

cat <<EOF

=============================================================================
 Xong! Các bước tiếp theo:
   conda activate ${ENV_NAME}
   bash scripts/move-kd/pretrain-v1.1.sh   # Stage 1: train MoLE + encoder adapters + projector
   bash scripts/move-kd/finetune-v1.1.sh   # Stage 2: train toàn bộ (trừ các teacher)
 Script mặc định dùng mọi GPU trên máy, batch 16/GPU (pre-train) và 8/GPU (fine-tune),
 tương ứng global batch 128/64 trên 8 GPU - nếu ít GPU hơn, tăng --gradient_accumulation_steps.
=============================================================================
EOF
