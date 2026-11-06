"""WER：词错误率，衡量匿名后文本内容保留程度（越低越好，论文 Table 2 用 WER）。

两套参考来源，二选一：
  1. 有真实标注：--ref_txt 目录里放与原始音频同名的 .txt（最严谨，推荐）；
  2. 无标注：用 Whisper 转写原始音频当作参考（近似）。

用法：
    python -m eval.wer --orig_dir data/orig --anon_dir outputs [--ref_txt data/ref_txt]
"""
import argparse
import os

import librosa
import whisper
from jiwer import wer


def transcribe_wav(model, path: str) -> str:
    """用 librosa 读成 16k 单声道 numpy 再转写，绕开 whisper 对 ffmpeg 子进程的依赖。"""
    sig, _ = librosa.load(path, sr=16000, mono=True)
    return model.transcribe(sig)["text"].strip()


def find_in_tree(root: str, filename: str) -> str | None:
    """在 root 下递归查找名为 filename 的文件（评测数据按说话人子目录组织）。"""
    for dirpath, _, files in os.walk(root):
        if filename in files:
            return os.path.join(dirpath, filename)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_dir", required=True, help="原始音频目录（含说话人子目录）")
    ap.add_argument("--anon_dir", required=True, help="匿名音频目录")
    ap.add_argument("--ref_txt", default=None, help="真实文本目录（可选，同名 .txt）")
    args = ap.parse_args()

    model = whisper.load_model("small.en")
    refs, hyps, names = [], [], []

    for fn in sorted(os.listdir(args.anon_dir)):
        if not fn.endswith(".wav"):
            continue
        # 命名约定：匿名文件 anon_<原名>.wav 对应原始 <原名>.wav
        stem = fn[len("anon_"):] if fn.startswith("anon_") else fn
        orig = find_in_tree(args.orig_dir, stem)
        if orig is None:
            print(f"  [skip] 找不到原始文件: {stem}")
            continue
        anon = os.path.join(args.anon_dir, fn)

        # 参考文本：优先真实标注，其次 Whisper 转写原始
        txt_path = os.path.join(args.ref_txt, stem.rsplit(".", 1)[0] + ".txt") if args.ref_txt else None
        if txt_path and os.path.isfile(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                ref = f.read().strip()
        else:
            ref = transcribe_wav(model, orig)

        hyp = transcribe_wav(model, anon)
        refs.append(ref)
        hyps.append(hyp)
        names.append(stem)

    if not refs:
        print("没有配对样本，请检查 --orig_dir / --anon_dir 的文件命名")
        return

    w = wer(refs, hyps)
    print(f"样本数: {len(refs)}  整体 WER: {w * 100:.2f}%")
    for i, name in enumerate(names):
        print(f"  {name}: ref='{refs[i][:60]}...' hyp='{hyps[i][:60]}...'")


if __name__ == "__main__":
    main()
