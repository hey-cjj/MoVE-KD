# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

MoVE-KD (CVPR 2025): distills several vision encoders (teachers: CLIP, EVA-02, ConvNeXt, SAM, optionally Pix2Struct) into a single CLIP ViT-L/14-336 student inside a LLaVA-1.5 VLM. It is a fork of LLaVA (Python package name is still `llava`) with encoder code borrowed from EAGLE. Most of the tree is upstream LLaVA; the MoVE-KD-specific code is concentrated in a few files (see below).

`Introduction.md` (Vietnamese) is a project write-up of the method (Sec 3.1–3.2) with a source-code mapping table in §2.4 and notes on where the code deviates from the paper. `setup.sh` (Vietnamese comments) is the environment setup script.

## Environment & commands

Training needs Linux + NVIDIA GPU (DeepSpeed + Flash-Attention 2). On this Windows checkout, code can be read/edited but not trained; use WSL2 or a Linux server. There is no test suite and no linter config.

```bash
bash setup.sh                    # conda env "move-kd", torch 2.1.2, deps, flash-attn, writes scripts/zero2*.json
DOWNLOAD_CKPT=1 bash setup.sh    # also fetch teacher checkpoints, CLIP, Vicuna
DOWNLOAD_DATA=1 bash setup.sh    # also fetch LCS-558K pretrain data + llava_v1_5_mix665k.json
SKIP_CONDA=1 bash setup.sh       # install into current Python (Docker/Colab)

bash scripts/move-kd/pretrain-v1.1.sh   # Stage 1
bash scripts/move-kd/finetune-v1.1.sh   # Stage 2 (reads Stage 1 outputs from ./checkpoints/pretrain/...)
```

- v1.0 vs v1.1 differ only in `--encoder_teachers` (`clip eva convnext` vs `clip sam eva convnext`) and output dir.
- Dependency pins beyond `pyproject.toml` are required (set in `setup.sh`): `timm==0.9.16` (ConvNeXt teacher needs `timm.layers`; ignore pip's timm==0.6.13 conflict warning), `numpy<2`, `peft==0.10.0`, plus `fvcore scipy tensorboard`.
- `*.json`/`*.jsonl` are gitignored, so `scripts/zero2.json` and `scripts/zero2_finetune.json` don't exist after clone; `setup.sh` generates them.
- GPUs below sm80 (V100/T4): use `llava/train/train.py` instead of `train_mem.py` (which forces `flash_attention_2`) and switch `--bf16/--tf32` to `--fp16 True --tf32 False`.
- Evaluation follows upstream LLaVA: `scripts/v1_5/eval/*.sh` (e.g. `python -m llava.eval.model_vqa_loader ...` then the matching `llava.eval.eval_*` scorer). These scripts still point at `liuhaotian/llava-v1.5-*`. Benchmark data lives under `playground/data/eval/`.
- MoVE-KD eval wrappers: `bash scripts/move-kd/eval/download_ckpt.sh` (saves HF weights under a `llava-*` dir name), then `[LIMIT=N] bash scripts/move-kd/eval/{pope,mme,textvqa,sqa}.sh [CKPT]`; `python scripts/move-kd/summarize_results.py --name <ckpt-dir-name>` builds tables/charts into `images/`. `scripts/move-kd/visualize_attention.py` draws [CLS]-attention / MoLE-routing maps from only the vision-tower shard (CPU OK). `scripts/move-kd/smoke_pretrain.sh` runs a ~30-step Stage 1 with TinyLlama. Write-up of results: `Experiments.md`; Colab A100 end-to-end runner: `notebooks/move_kd_colab.ipynb` (generated, edit cells directly).
- Don't eval stock LLaVA-1.5 checkpoints with this repo: `builder.py` loads every `llava`-named model as `EvalMoVELlavaLlamaForCausalLM`, whose config defaults `moe_encoder=True` (random LoRA experts get attached to CLIP), and the `kd_mode=False` path is broken (`encode_images` always unpacks two values from the vision tower). Use `scripts/move-kd/eval/baseline.sh`, which runs the same wrappers from an upstream LLaVA checkout (symlinked `playground/`); `PYTHONPATH` alone doesn't work because `python -m` puts the cwd first.
- `.gitattributes` forces LF for `*.sh` so scripts run under bash from a Windows checkout; keep it that way.

## Architecture (MoVE-KD-specific parts)

**Model class** — `llava/model/language_model/move_llava_llama.py`
- `MoVELlavaLlamaForCausalLM` (config type `move_llava_llama`) is what `train.py` instantiates for training (the plain `LlavaLlamaForCausalLM` path is commented out).
- `initialize_teachers()` builds the teachers, replaces every CLIP encoder layer's FFN with a `MoE_LoRA_CLIP` block (Mixture-of-LoRA-Experts, "MoLE"), and creates per-teacher **encoder adapters** that project each teacher's features to the student's 576×1024 token grid.
- `forward()` = LM loss + `moe_encoder_balance_w` × router load-balancing loss + `kd_w` × KD loss. KD is per-teacher MSE between student tokens and adapted teacher tokens, optionally re-weighted per token (`--token_weight`, CLS attention) and per teacher (`--teacher_weight`, similarity to CLS); the CLIP teacher gets fixed weight `kd_memory_w` (0.8).
- `EvalMoVELlavaLlamaForCausalLM` is the inference variant: rebuilds the MoLE structure from `config.moe` and has no teachers. `llava/model/builder.py` loads it only when the model name contains `llava` (and not `mpt`/`mistral`). The name comes from the last path component (`mm_utils.get_model_name_from_path`), so a checkpoint dir like `checkpoints/finetune/move-kd-7b-v1.1` or HF `jiaojuncao/MoVE-KD-7b-v1.1` falls through to the plain-LLM branch unless the directory is renamed to include `llava`.

**MoLE** — `llava/model/multimodal_encoder/moe_lora.py`: `LoRALayer`, `MoE_LoRA_CLIP` (router + LoRA experts on top of the frozen original MLP; sparse or `--moe_encoder_dense`), and `MoECLIP*` encoder/layer/transformer subclasses that additionally return routing info used by the balance loss.

**Teachers** — `llava/model/multimodal_encoder/mixture_vision_models.py::init_vision_teachers` with implementations in `multimodal_encoder/vision_models/`. Key facts:
- Teacher checkpoint paths are **hardcoded relative to repo root**: `./checkpoints/sam/sam_vit_l`, `./checkpoints/EVA/eva02/det/eva02_L_coco_det_sys_o365.pth`, `./checkpoints/pix2struct`; ConvNeXt (`convnext_xxlarge.clip_laion2b_soup`) is fetched via timm. Run training from repo root.
- Teachers are stored in a plain Python `dict` (`self.vision_teachers`), not as submodules, so they never appear in `model.parameters()`, are never trained (even with `--tune_all`), and are never saved in checkpoints.

**Training script** — `llava/train/train.py`
- MoVE-KD flags live in `ModelArguments` (`encoder_teachers`, `moe_encoder*`, `pretrain_mole_encoder`, `pretrain_encoder_adapter`) and `TrainingArguments` (`finetune`, `tune_encoder`, `tune_encoder_adapter`, `tune_all`, `kd`, `kd_w`, `kd_memory_w`, `token_weight`, `teacher_weight`, `moe_encoder_balance_w`).
- Stage 1 (`--tune_mm_mlp_adapter`, not `--finetune`): trains MoLE + encoder adapters + projector. `safe_save_model_for_hf_trainer` writes `mm_projector.bin`, `mole_encoder.bin`, `encoder_adapter.bin` into the output dir.
- Stage 2 (`--finetune --tune_all`): loads those three files via `--pretrain_mm_mlp_adapter`, `--pretrain_mole_encoder`, `--pretrain_encoder_adapter` and trains everything except teachers.

**Implementation quirks worth knowing** (from `Introduction.md` §2.4): in sparse MoLE mode the router only gets gradient from the balance loss (expert outputs aren't scaled by router probs; straight-through choice is detached); LoRA `B` is not zero-initialized (both A and B use Kaiming-uniform), so MoLE output differs slightly from the original FFN at init.
