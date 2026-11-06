"""从 RAVDESS 数据集批量计算 4 类情感 F0 质心，输出 emotion_prototypes.npy。

论文 (Lee et al. 2024) 用 IEMOCAP 训练集按情感类求 F0 质心作为 prototype。
这里用公开的 RAVDESS 替代（IEMOCAP 需申请才能下载），方法完全一致：
  1. 递归扫描 data_dir 下所有 *.wav；
  2. 从文件名解析情感标签（RAVDESS 文件名第 5 段）；
  3. 对每个音频用 librosa.pyin 提 F0（与 app/emotion.py 同参数），
     只统计有声帧（F0>0）的均值；
  4. 按情感类求平均 → 各类的 F0 质心。

RAVDESS 文件名格式（示例 03-01-01-01-01-01-01.wav）：
  03(模态=01音频) - 01(声道=01语音) - 01(情感) - 01(强度) - 01(语句) - 01(重复) - 01(演员)
  情感编码: 01=neutral 02=calm 03=happy 04=sad 05=angry 06=fearful 07=disgust 08=surprised
论文用 4 类：neutral / happy / sad / angry（映射见 EMOTION_MAP）。

用法:
    python scripts/compute_emotion_prototypes.py --data_dir D:/RAVDESS --out data/emotion_prototypes.npy
"""
import argparse
import os
import re
import sys

import numpy as np

# 把项目根目录加入 sys.path，复用 app/emotion.py 的 extract_f0
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.emotion import extract_f0  # noqa: E402

# RAVDESS 情感编码 → 论文 4 类（只取论文用到的 4 个情感类）
EMOTION_MAP = {
    "01": "neu",  # neutral
    "03": "hap",  # happy
    "04": "sad",  # sad
    "05": "ang",  # angry
}
PROTO_ORDER = ["neu", "hap", "sad", "ang"]  # 输出顺序，与 app/pipeline.py 默认一致


def parse_emotion(filename: str) -> str | None:
    """从 RAVDESS 文件名第 5 段解析情感类，返回 'neu'/'hap'/'sad'/'ang' 或 None。"""
    # 文件名如 03-01-01-01-01-01-01.wav，按 '-' 拆
    parts = re.match(r"(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)", filename)
    if not parts:
        return None
    emotion_code = parts.group(3)
    return EMOTION_MAP.get(emotion_code)


def compute_prototypes(data_dir: str) -> dict:
    """扫描 data_dir 下所有 wav，按情感类累计 F0 均值，返回 {情感类: 质心}。"""
    sums: dict[str, list[float]] = {k: [] for k in PROTO_ORDER}
    n_parsed = n_skipped = 0
    for root, _, files in os.walk(data_dir):
        for fn in sorted(files):
            if not fn.lower().endswith(".wav"):
                continue
            emo = parse_emotion(fn)
            if emo is None:
                n_skipped += 1
                continue
            n_parsed += 1
            path = os.path.join(root, fn)
            try:
                f0 = extract_f0(path)
            except Exception as e:  # noqa: BLE001 个别文件坏了不影响整体
                print(f"  [warn] {fn}: {e}")
                continue
            voiced = f0[f0 > 0]
            if voiced.size == 0:
                continue  # 无声段
            sums[emo].append(float(np.mean(voiced)))

    print(f"解析音频: {n_parsed}，跳过(非目标情感/命名不符): {n_skipped}")
    prototypes = {}
    for emo in PROTO_ORDER:
        vals = sums[emo]
        if vals:
            prototypes[emo] = float(np.mean(vals))
            print(f"  {emo:>4}: 样本数={len(vals):4d}  F0质心={prototypes[emo]:7.2f} Hz")
        else:
            print(f"  {emo:>4}: 无样本，质心置 0（会在 pipeline 中被警告）")
            prototypes[emo] = 0.0
    return prototypes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", required=True, help="RAVDESS 根目录（递归扫描 wav）")
    ap.add_argument("--out", default="data/emotion_prototypes.npy",
                    help="输出 npy 路径（默认 data/emotion_prototypes.npy）")
    args = ap.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"[error] 目录不存在: {args.data_dir}")
        sys.exit(1)

    prototypes = compute_prototypes(args.data_dir)
    arr = np.array([prototypes[k] for k in PROTO_ORDER], dtype=np.float32)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, arr)
    print(f"\n已保存到 {args.out}: {arr}")


if __name__ == "__main__":
    main()
