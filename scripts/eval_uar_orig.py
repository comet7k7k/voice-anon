"""原始音频 UAR 基线：在 RAVDESS 原始音频上评测 SER 模型的情感识别性能。

与 eval/uar.py 使用同一 SER 模型（superb/wav2vec2-base-superb-er）与同一真实标签
来源（RAVDESS 文件名情感码），得到"原始"列的对照值，供报告 6.2 节与论文 Orig 对比。

用法：
    python scripts/eval_uar_orig.py --orig_dir data/orig
"""
import argparse
import os
import re

import numpy as np

LABELS = ["ang", "hap", "sad", "neu"]
RAVDESS_EMO_MAP = {"01": "neu", "03": "hap", "04": "sad", "05": "ang"}


def ravdess_label(filename: str) -> str | None:
    m = re.match(r"(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)", filename)
    if not m or m.group(1) != "03" or m.group(2) != "01":
        return None
    return RAVDESS_EMO_MAP.get(m.group(3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_dir", required=True, help="原始音频目录（含说话人子目录）")
    args = ap.parse_args()

    import librosa
    import torch
    from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

    ser_name = "superb/wav2vec2-base-superb-er"
    print(f"[uar-orig] 加载 SER 模型 {ser_name} ...")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(ser_name)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(ser_name)

    def norm_label(lab: str) -> str:
        lab = lab.lower()
        for k in LABELS:
            if lab.startswith(k):
                return k
        return lab[:3]

    def predict(path: str) -> str:
        sig, _ = librosa.load(path, sr=16000, mono=True)
        inputs = feature_extractor(sig, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        idx = int(logits.argmax(-1).item())
        return norm_label(model.config.id2label.get(idx, str(idx)))

    correct, total, n = {}, {}, 0
    for spk in sorted(os.listdir(args.orig_dir)):
        spk_dir = os.path.join(args.orig_dir, spk)
        if not os.path.isdir(spk_dir):
            continue
        for fn in sorted(os.listdir(spk_dir)):
            if not fn.endswith(".wav"):
                continue
            truth = ravdess_label(fn)
            if truth is None:
                print(f"  [skip] 非 RAVDESS 命名: {fn}")
                continue
            pred = predict(os.path.join(spk_dir, fn))
            n += 1
            total[truth] = total.get(truth, 0) + 1
            if pred == truth:
                correct[truth] = correct.get(truth, 0) + 1
            print(f"  {fn}: truth={truth} pred={pred} {'✓' if pred == truth else '✗'}")

    if n == 0:
        print("没有可评测样本"); return

    recalls = []
    for lab in LABELS:
        t = total.get(lab, 0)
        c = correct.get(lab, 0)
        r = c / t if t else 0.0
        recalls.append(r)
        print(f"  {lab:>3}: 样本数={t:3d}  召回率={r * 100:6.2f}%")
    uar = float(np.mean(recalls))
    print(f"原始音频 UAR（无加权平均召回率）: {uar * 100:.2f}%")


if __name__ == "__main__":
    main()
