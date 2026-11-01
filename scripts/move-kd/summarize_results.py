"""Tổng hợp kết quả MoVE-KD: số liệu paper (Table 1-5) + kết quả tự chạy -> bảng markdown & biểu đồ.

Số liệu paper chép từ arXiv:2501.01709 (CVPR 2025). Kết quả tự chạy được đọc từ các file
*_score.txt mà scripts/move-kd/eval/*.sh ghi ra.

Ví dụ (chạy ở thư mục gốc repo):
    python scripts/move-kd/summarize_results.py                       # chỉ số liệu paper
    python scripts/move-kd/summarize_results.py --name llava-move-kd-7b-v1.1
"""
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EVAL_DIR = os.path.join(REPO_ROOT, "playground", "data", "eval")

BENCHMARKS = ["VQAv2", "GQA", "TextVQA", "VizWiz", "POPE", "SQA", "MME", "MMB"]

# Table 1 (các dòng LLaVA-1.5); cùng thứ tự BENCHMARKS
TABLE1 = {
    "7B": {
        "LLaVA-1.5": [78.5, 62.0, 58.2, 50.0, 85.9, 66.8, 1510.7, 64.3],
        "MoVE-KD-v1.0": [79.5, 63.2, 58.3, 52.3, 86.9, 69.3, 1524.5, 66.3],
        "MoVE-KD-v1.1": [79.9, 63.9, 59.6, 52.7, 86.3, 69.8, 1509.1, 67.4],
    },
    "13B": {
        "LLaVA-1.5": [80.0, 63.3, 61.3, 53.6, 85.9, 71.6, 1531.3, 67.7],
        "MoVE-KD-v1.0": [80.6, 64.2, 59.7, 55.7, 85.7, 73.2, 1568.1, 70.2],
        "MoVE-KD-v1.1": [80.8, 63.9, 61.1, 57.5, 86.3, 71.8, 1568.3, 69.7],
    },
}

# Table 2: thêm lần lượt từng thành phần (LLaVA-1.5 7B); Avg của paper = trung bình 7 benchmark, không tính MME
TABLE2 = [
    ("LLaVA-1.5 7B", [78.5, 62.0, 58.2, 50.0, 85.9, 66.8, 1510.7, 64.3], 66.5),
    ("+ KD (nội suy)", [79.0, 62.4, 56.7, 50.9, 84.7, 67.6, 1507.6, 62.9], 66.3),
    ("+ Encoder adapter", [79.3, 62.4, 57.0, 51.2, 85.2, 68.3, 1517.8, 63.8], 66.7),
    ("+ MoLE", [79.1, 62.8, 57.3, 51.9, 86.4, 68.7, 1521.3, 65.4], 67.4),
    ("+ Token weight", [79.3, 63.1, 57.7, 52.5, 86.7, 68.9, 1524.1, 66.0], 67.7),
    ("+ Teacher weight", [79.5, 63.2, 58.3, 52.3, 86.9, 69.3, 1524.5, 66.3], 68.0),
]

# Table 4: trọng số teacher CLIP (kd_memory_w)
TABLE4 = [
    (0.6, [78.8, 62.6, 57.3, 51.7, 86.6, 68.2, 1518.4, 64.5], 67.1),
    (0.7, [79.1, 62.5, 57.9, 52.1, 86.3, 69.5, 1521.7, 65.2], 67.5),
    (0.8, [79.5, 63.2, 58.3, 52.3, 86.9, 69.3, 1524.5, 66.3], 68.0),
    (0.9, [80.1, 62.9, 57.8, 51.9, 86.6, 69.4, 1519.5, 65.8], 67.8),
]

# Palette categorical đã kiểm định (thứ tự cố định) + màu chữ/lưới trung tính
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join([" :--- "] + [" ---: "] * (len(header) - 1)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def fmt(v):
    return "-" if v is None else (f"{v:.1f}" if isinstance(v, float) else str(v))


def read_scores(name):
    """Đọc *_score.txt do scripts/move-kd/eval/*.sh ghi ra. Trả về {benchmark: điểm} (thang như paper)."""
    def read(*parts):
        path = os.path.join(EVAL_DIR, *parts)
        return open(path, encoding="utf-8").read() if os.path.isfile(path) else None

    scores = {}
    text = read("pope", "answers", f"{name}_score.txt")
    if text:  # POPE: F1 trung bình 3 split (adversarial/popular/random), như LLaVA-1.5
        f1 = [float(x) for x in re.findall(r"F1 score: ([\d.]+)", text)]
        if f1:
            scores["POPE"] = 100 * sum(f1) / len(f1)
    text = read("textvqa", "answers", f"{name}_score.txt")
    if text and (m := re.search(r"Accuracy: ([\d.]+)%", text)):
        scores["TextVQA"] = float(m.group(1))
    text = read("scienceqa", "answers", f"{name}_score.txt")
    if text and (m := re.search(r"IMG-Accuracy: ([\d.]+)%", text)):  # paper báo cáo SQA-IMG
        scores["SQA"] = float(m.group(1))
    text = read("MME", "answers", f"{name}_score.txt")
    if text and (m := re.search(r"Perception =+\s*total score: ([\d.]+)", text)):  # paper báo cáo Perception
        scores["MME"] = float(m.group(1))
    return scores


def plot_main(out):
    """Thay đổi tương đối (%) so với LLaVA-1.5 - một trục chung cho mọi benchmark (MME khác thang đo)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    x = np.arange(len(BENCHMARKS))
    width = 0.38
    for ax, (size, rows) in zip(axes, TABLE1.items()):
        base = np.array(rows["LLaVA-1.5"])
        for i, variant in enumerate(["MoVE-KD-v1.0", "MoVE-KD-v1.1"]):
            delta = (np.array(rows[variant]) - base) / base * 100
            ax.bar(x + (i - 0.5) * width, delta, width - 0.04, color=SERIES[i], label=variant)
        ax.axhline(0, color=INK_2, linewidth=0.8)
        ax.set_xticks(x, BENCHMARKS)
        ax.set_title(f"Vicuna-{size}", color=INK, fontsize=11)
        style(ax)
    axes[0].set_ylabel("Thay đổi so với LLaVA-1.5 (%)", color=INK_2)
    axes[1].legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("Table 1: MoVE-KD so với LLaVA-1.5 (số liệu paper)", color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_ablation(out):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    labels = [r[0] for r in TABLE2]
    avg = [r[2] for r in TABLE2]
    y = np.arange(len(labels))[::-1]
    ax.barh(y, avg, height=0.6, color=SERIES[0])
    for yi, v in zip(y, avg):
        ax.text(v + 0.05, yi, f"{v:.1f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, labels)
    ax.set_xlim(65.5, 68.5)
    ax.set_xlabel("Avg 7 benchmark (không tính MME)", color=INK_2)
    ax.set_title("Table 2: thêm lần lượt từng thành phần của MoVE-KD", color=INK, fontsize=11)
    style(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_clip_weight(out):
    fig, ax = plt.subplots(figsize=(5, 3.4))
    w = [r[0] for r in TABLE4]
    avg = [r[2] for r in TABLE4]
    ax.plot(w, avg, color=SERIES[0], linewidth=2, marker="o", markersize=8)
    best = int(np.argmax(avg))
    ax.annotate(f"{avg[best]:.1f} (mặc định)", (w[best], avg[best]), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=9, color=INK)
    ax.set_ylim(min(avg) - 0.2, max(avg) + 0.3)
    ax.set_xticks(w)
    ax.set_xlabel("Trọng số teacher CLIP (--kd_memory_w)", color=INK_2)
    ax.set_ylabel("Avg (không tính MME)", color=INK_2)
    ax.set_title("Table 4: ảnh hưởng của trọng số CLIP", color=INK, fontsize=11)
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_reproduced(paper, ours, name, out):
    keys = [k for k in BENCHMARKS if k in ours]
    x = np.arange(len(keys))
    # Tương đối (%) để MME (thang ~1500) và các benchmark thang 100 dùng chung một trục
    delta = [(ours[k] - paper[BENCHMARKS.index(k)]) / paper[BENCHMARKS.index(k)] * 100 for k in keys]
    fig, ax = plt.subplots(figsize=(1.6 * len(keys) + 2, 3.4))
    ax.bar(x, delta, 0.5, color=SERIES[0])
    for xi, k, d in zip(x, keys, delta):
        ax.text(xi, d, f"{ours[k]:.1f}\n(paper {paper[BENCHMARKS.index(k)]:.1f})", ha="center",
                va="bottom" if d >= 0 else "top", fontsize=8, color=INK)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_xticks(x, keys)
    ax.set_ylabel("Tự chạy so với paper (%)", color=INK_2)
    ax.set_title(f"Kết quả tái lập: {name}", color=INK, fontsize=11)
    style(ax)
    lim = max(1.0, max(abs(d) for d in delta) * 1.8)
    ax.set_ylim(-lim, lim)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", nargs="*", default=[],
                        help="tên checkpoint đã eval (vd. llava-move-kd-7b-v1.1, thêm _limitN nếu chạy LIMIT)")
    parser.add_argument("--paper-ref", default="MoVE-KD-v1.1", choices=["MoVE-KD-v1.0", "MoVE-KD-v1.1", "LLaVA-1.5"],
                        help="dòng Table 1 (7B) để so sánh với kết quả tự chạy")
    parser.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "images"))
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    plot_main(os.path.join(args.out_dir, "results_table1.png"))
    plot_ablation(os.path.join(args.out_dir, "results_ablation.png"))
    plot_clip_weight(os.path.join(args.out_dir, "results_clip_weight.png"))
    print(f"[ OK ] results_table1.png, results_ablation.png, results_clip_weight.png -> {args.out_dir}")

    paper = TABLE1["7B"][args.paper_ref]
    for name in args.name:
        ours = read_scores(name)
        if not ours:
            print(f"[WARN] Không tìm thấy *_score.txt cho '{name}' trong {EVAL_DIR}")
            continue
        rows = [[k, fmt(paper[BENCHMARKS.index(k)]), fmt(ours[k]), f"{ours[k] - paper[BENCHMARKS.index(k)]:+.1f}"]
                for k in BENCHMARKS if k in ours]
        print(f"\n### {name} vs paper ({args.paper_ref} 7B)\n")
        print(md_table(["Benchmark", "Paper", "Tự chạy", "Chênh lệch"], rows))
        plot_reproduced(paper, ours, name, os.path.join(args.out_dir, f"results_reproduced_{name}.png"))


if __name__ == "__main__":
    main()
