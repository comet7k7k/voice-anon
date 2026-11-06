"""原始音频 WER 基线：Whisper 转写 RAVDESS 原始音频，与官方固定文本比对。

与 eval/wer.py 使用同一 Whisper 模型（small.en）与同一参考文本（data/ref_txt），
得到"原始"列的对照值，供报告 6.2 节与匿名后 WER 对比。

用法：
    python scripts/eval_wer_orig.py --orig_dir data/orig --ref_txt data/ref_txt
"""
import argparse
import os

import librosa
import whisper
from jiwer import wer


def transcribe_wav(model, path: str) -> str:
    sig, _ = librosa.load(path, sr=16000, mono=True)
    return model.transcribe(sig)["text"].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_dir", required=True, help="原始音频目录（含说话人子目录）")
    ap.add_argument("--ref_txt", required=True, help="真实文本目录（与 wav 同名 .txt）")
    args = ap.parse_args()

    model = whisper.load_model("small.en")
    refs, hyps, names = [], [], []
    for spk in sorted(os.listdir(args.orig_dir)):
        spk_dir = os.path.join(args.orig_dir, spk)
        if not os.path.isdir(spk_dir):
            continue
        for fn in sorted(os.listdir(spk_dir)):
            if not fn.endswith(".wav"):
                continue
            stem = fn.rsplit(".", 1)[0]
            txt = os.path.join(args.ref_txt, stem + ".txt")
            if not os.path.isfile(txt):
                print(f"  [skip] 缺标注: {fn}")
                continue
            with open(txt, "r", encoding="utf-8") as f:
                ref = f.read().strip()
            hyp = transcribe_wav(model, os.path.join(spk_dir, fn))
            refs.append(ref)
            hyps.append(hyp)
            names.append(stem)

    if not refs:
        print("没有配对样本"); return
    w = wer(refs, hyps)
    print(f"样本数: {len(refs)}  原始音频 WER: {w * 100:.2f}%")
    for i, name in enumerate(names):
        print(f"  {name}: ref='{refs[i][:60]}...' hyp='{hyps[i][:60]}...'")


if __name__ == "__main__":
    main()
