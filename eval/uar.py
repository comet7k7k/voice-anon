"""UAR：情感识别无加权平均召回率，衡量匿名后情感保留程度（越高越好，论文 Table 3 用 UAR）。

用 SUPERB 的 wav2vec2-base-superb-er 情感识别模型（transformers 加载，走 HF 镜像下载）。
该模型在 IEMOCAP 上微调，输出 4 类情感：ang / hap / sad / neu，与论文 4 类一致。

UAR = 各类别召回率的算术平均（无加权），与论文口径一致。
真实标签来源：优先解析 RAVDESS 文件名第 5 段（01=neu 03=hap 04=sad 05=ang）；
非 RAVDESS 命名时回退为"原始音频的 SER 预测"（近似，见 README 局限）。

用法：
    python -m eval.uar --orig_dir data/orig --anon_dir outputs
"""
import argparse
import os
import re

import numpy as np

# IEMOCAP 情感标签（SUPERB ER 模型输出顺序，与论文 4 类一致）
LABELS = ["ang", "hap", "sad", "neu"]
# RAVDESS 文件名第 5 段情感码 → 4 类
RAVDESS_EMO_MAP = {"01": "neu", "03": "hap", "04": "sad", "05": "ang"}


def find_in_tree(root: str, filename: str) -> str | None:
    """在 root 下递归查找名为 filename 的文件（评测数据按说话人子目录组织）。"""
    for dirpath, _, files in os.walk(root):
        if filename in files:
            return os.path.join(dirpath, filename)
    return None


def ravdess_label(filename: str) -> str | None:
    """从 RAVDESS 文件名解析真实情感标签；非 RAVDESS 命名返回 None。"""
    m = re.match(r"(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)", filename)
    if not m or m.group(1) != "03" or m.group(2) != "01":
        return None
    return RAVDESS_EMO_MAP.get(m.group(3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_dir", required=True, help="原始音频目录（含真实情感标签的音频）")
    ap.add_argument("--anon_dir", required=True, help="匿名音频目录")
    args = ap.parse_args()

    # 懒加载 SER 模型：首次运行会从 HF 镜像下载（约 240MB）
    import librosa
    import torch
    from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

    ser_name = "superb/wav2vec2-base-superb-er"
    print(f"[uar] 加载 SER 模型 {ser_name} ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Wav2Vec2ForSequenceClassification.from_pretrained(ser_name).to(device)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(ser_name)

    # id2label 可能是全名（angry/happy/sad/neutral），统一归一到 3 字母
    def norm_label(lab: str) -> str:
        lab = lab.lower()
        for k in LABELS:
            if lab.startswith(k):
                return k
        return lab[:3]

    def predict(path: str) -> str:
        sig, _ = librosa.load(path, sr=16000, mono=True)
        inputs = feature_extractor(sig, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        idx = int(logits.argmax(-1).item())
        return norm_label(model.config.id2label.get(idx, str(idx)))

    # 配对所有 (原始, 匿名)
    correct = {}
    total = {}
    n_paired = 0
    for fn in sorted(os.listdir(args.anon_dir)):
        if not fn.endswith(".wav"):
            continue
        stem = fn[len("anon_"):] if fn.startswith("anon_") else fn
        orig = find_in_tree(args.orig_dir, stem)
        anon = os.path.join(args.anon_dir, fn)
        if orig is None:
            continue
        n_paired += 1
        # 真实标签：优先 RAVDESS 文件名解析，回退到原始音频的 SER 预测
        y_orig = ravdess_label(stem) or predict(orig)
        y_anon = predict(anon)
        total[y_orig] = total.get(y_orig, 0) + 1
        if y_orig == y_anon:
            correct[y_orig] = correct.get(y_orig, 0) + 1
        print(f"  {stem}: orig={y_orig} anon={y_anon} {'✓' if y_orig == y_anon else '✗'}")

    if n_paired == 0:
        print("没有配对样本"); return

    recalls = []
    for lab in LABELS:
        n = total.get(lab, 0)
        c = correct.get(lab, 0)
        r = c / n if n else 0.0
        recalls.append(r)
        print(f"  {lab:>3}: 样本数={n:3d}  召回率={r * 100:6.2f}%")
    uar = float(np.mean(recalls))
    print(f"UAR（无加权平均召回率）: {uar * 100:.2f}%")


if __name__ == "__main__":
    main()
