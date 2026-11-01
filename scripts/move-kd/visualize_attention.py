"""Trực quan hoá [CLS] attention và routing của MoLE (tương tự Fig 5 của paper MoVE-KD).

Chỉ dùng vision encoder (CLIP ViT-L/14-336), không nạp LLM 7B, nên chạy được cả trên CPU:
  * CLIP gốc   : openai/clip-vit-large-patch14-336 (cũng là teacher "neo" khi KD)
  * Student    : CLIP + MoLE lấy từ checkpoint MoVE-KD (chỉ tải shard chứa vision tower)

Cách tính token weight giống hệt MoVELlavaLlamaForCausalLM.forward (move_llava_llama.py):
    attn = attentions[mm_vision_select_layer].mean(heads)[:, 0, 1:]   # [CLS] -> 576 patch
    w    = softmax(attn)                                              # lưới 24x24

Ví dụ (chạy ở thư mục gốc repo):
    python scripts/move-kd/visualize_attention.py
    python scripts/move-kd/visualize_attention.py --images a.jpg b.jpg --ckpt checkpoints/llava-move-kd-7b-v1.1
"""
import argparse
import importlib.util
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CLIP_NAME = "openai/clip-vit-large-patch14-336"
VT_PREFIX = "model.vision_tower.vision_tower."  # tiền tố của vision tower trong checkpoint MoVE-KD


def load_moe_lora_module():
    # Nạp trực tiếp moe_lora.py (chỉ cần transformers) thay vì 'import llava',
    # vì llava/__init__ kéo theo toàn bộ teacher encoder (timm, fvcore, ...).
    path = os.path.join(REPO_ROOT, "llava", "model", "multimodal_encoder", "moe_lora.py")
    spec = importlib.util.spec_from_file_location("moe_lora", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_ckpt_file(ckpt, filename):
    """Lấy file từ thư mục checkpoint local, hoặc tải từ HF Hub nếu ckpt là repo id."""
    local = os.path.join(ckpt, filename)
    if os.path.isfile(local):
        return local
    return hf_hub_download(ckpt, filename)


def build_student(ckpt, device, dtype):
    """Dựng CLIP + MoLE theo đúng trình tự EvalMoVELlavaLlamaForCausalLM.__init__ rồi nạp trọng số."""
    moe_lora = load_moe_lora_module()
    moe_cfg = json.load(open(read_ckpt_file(ckpt, "config.json")))["moe"]

    # Toàn bộ trọng số (kể cả phần CLIP gốc) lấy từ checkpoint nên chỉ cần config
    config = CLIPVisionConfig.from_pretrained(CLIP_NAME)
    model = CLIPVisionModel(config)
    model.vision_model = moe_lora.MoECLIPVisionTransformer(config)
    model.vision_model.encoder = moe_lora.MoECLIPEncoder(config)
    for i in range(len(model.vision_model.encoder.layers)):
        layer = moe_lora.MoECLIPEncoderLayer(config)
        layer.mlp = moe_lora.MoE_LoRA_CLIP(
            args=None,
            lora_rank=moe_cfg["moe_encoder_lora_rank"],
            lora_alpha=moe_cfg["moe_encoder_lora_alpha"],
            num_experts=moe_cfg["moe_encoder_num_experts"],
            dense_moe=moe_cfg["moe_encoder_dense"],
            original_module=layer.mlp,
        )
        model.vision_model.encoder.layers[i] = layer

    weight_map = json.load(open(read_ckpt_file(ckpt, "model.safetensors.index.json")))["weight_map"]
    shards = sorted({f for k, f in weight_map.items() if k.startswith(VT_PREFIX)})
    state_dict = {}
    for shard in shards:
        with safe_open(read_ckpt_file(ckpt, shard), framework="pt") as f:
            for key in f.keys():
                if key.startswith(VT_PREFIX):
                    state_dict[key[len(VT_PREFIX):]] = f.get_tensor(key)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # position_ids là buffer không được lưu trong checkpoint, bỏ qua
    missing = [k for k in missing if not k.endswith("position_ids")]
    if missing or unexpected:
        raise RuntimeError(f"Nạp vision tower lỗi: missing={missing[:5]} unexpected={unexpected[:5]}")
    print(f"[ OK ] Student MoLE: {len(state_dict)} tensor từ {shards}")
    return model.to(device=device, dtype=dtype).eval(), moe_cfg["moe_encoder_num_experts"]


def expand2square(img, background_color):
    # Giống MoVELlavaLlamaForCausalLM.process_images (image_aspect_ratio = pad)
    w, h = img.size
    if w == h:
        return img
    size = max(w, h)
    result = Image.new(img.mode, (size, size), background_color)
    result.paste(img, ((size - w) // 2, (size - h) // 2))
    return result


@torch.no_grad()
def cls_attention(model, pixel_values, select_layer):
    out = model.vision_model(pixel_values, output_attentions=True, output_hidden_states=True, return_dict=True)
    attn = out.attentions[select_layer].float().mean(1)  # [b, 577, 577]
    weight = torch.softmax(attn[:, 0, 1:], dim=-1)  # [b, 576]
    side = int(weight.shape[-1] ** 0.5)
    return weight.view(-1, side, side), getattr(out, "routings", None)


def overlay(ax, image, heat, title):
    heat = F.interpolate(heat[None, None], size=image.size[::-1], mode="bilinear", align_corners=False)[0, 0]
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    ax.imshow(image)
    ax.imshow(heat.numpy(), cmap="jet", alpha=0.5)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", default="jiaojuncao/MoVE-KD-7b-v1.1",
                        help="thư mục checkpoint local hoặc HF repo id (chỉ tải shard chứa vision tower)")
    parser.add_argument("--images", nargs="+",
                        default=[os.path.join(REPO_ROOT, "llava/serve/examples", f)
                                 for f in ("extreme_ironing.jpg", "waterview.jpg")])
    parser.add_argument("--select-layer", type=int, default=-2, help="mm_vision_select_layer của MoVE-KD")
    parser.add_argument("--routing-layers", type=int, nargs="+", default=[0, 8, 16, 22])
    parser.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "images"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    os.makedirs(args.out_dir, exist_ok=True)

    processor = CLIPImageProcessor.from_pretrained(CLIP_NAME)
    clip = CLIPVisionModel.from_pretrained(CLIP_NAME, torch_dtype=dtype).to(device).eval()
    student, num_experts = build_student(args.ckpt, device, dtype)
    background = tuple(int(x * 255) for x in processor.image_mean)

    expert_share = np.zeros((len(student.vision_model.encoder.layers), num_experts))
    for path in args.images:
        name = os.path.splitext(os.path.basename(path))[0]
        image = expand2square(Image.open(path).convert("RGB"), background)
        pixel_values = processor(image, return_tensors="pt")["pixel_values"].to(device, dtype)

        clip_map, _ = cls_attention(clip, pixel_values, args.select_layer)
        student_map, routings = cls_attention(student, pixel_values, args.select_layer)

        # 1) [CLS] attention: CLIP gốc vs student MoVE-KD
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
        axes[0].imshow(image); axes[0].set_title("Ảnh (pad vuông)", fontsize=10); axes[0].axis("off")
        overlay(axes[1], image, clip_map[0].cpu(), "CLIP ViT-L/14-336")
        overlay(axes[2], image, student_map[0].cpu(), "Student MoVE-KD")
        fig.suptitle(f"[CLS] attention, layer {args.select_layer} (token weight của KD)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"attn_{name}.png"), dpi=150)
        plt.close(fig)

        # 2) Routing của MoLE: expert được chọn cho từng patch token
        experts = torch.stack([r[0][0, 1:].argmax(-1) for r in routings]).cpu()  # [layers, 576]
        for e in range(num_experts):
            expert_share[:, e] += (experts == e).float().mean(1).numpy()
        side = int(experts.shape[1] ** 0.5)
        cmap = plt.get_cmap("tab10", num_experts)
        fig, axes = plt.subplots(1, len(args.routing_layers) + 1, figsize=(3.2 * (len(args.routing_layers) + 1), 3.4))
        axes[0].imshow(image.resize((336, 336))); axes[0].set_title("Ảnh", fontsize=10); axes[0].axis("off")
        for ax, layer in zip(axes[1:], args.routing_layers):
            ax.imshow(experts[layer].view(side, side).numpy(), cmap=cmap, vmin=-0.5, vmax=num_experts - 0.5)
            ax.set_title(f"Layer {layer}", fontsize=10)
            ax.axis("off")
        handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(e)) for e in range(num_experts)]
        fig.legend(handles, [f"Expert {e}" for e in range(num_experts)], loc="lower center", ncol=num_experts)
        fig.suptitle("Expert MoLE được router chọn cho mỗi patch token")
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(os.path.join(args.out_dir, f"routing_{name}.png"), dpi=150)
        plt.close(fig)
        print(f"[ OK ] {name}: attn_{name}.png, routing_{name}.png")

    # 3) Tỉ lệ token đi vào mỗi expert theo layer (trung bình trên các ảnh)
    expert_share /= len(args.images)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    bottom = np.zeros(expert_share.shape[0])
    for e in range(num_experts):
        ax.bar(np.arange(expert_share.shape[0]), expert_share[:, e], bottom=bottom, label=f"Expert {e}")
        bottom += expert_share[:, e]
    ax.set_xlabel("Layer của student encoder")
    ax.set_ylabel("Tỉ lệ patch token")
    ax.set_title(f"Phân bố routing MoLE ({len(args.images)} ảnh)")
    ax.legend(ncol=num_experts, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "routing_share.png"), dpi=150)
    print(f"[ OK ] routing_share.png -> {args.out_dir}")


if __name__ == "__main__":
    main()
