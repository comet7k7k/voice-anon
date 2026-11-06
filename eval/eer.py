"""EER：等错误率，衡量匿名强度（越高越难识别原说话人 = 匿名越强，论文 Table 1 用 EER）。

数据组织（推荐按说话人建子目录，才能构造正负样本对）：
    orig_dir/
      spk01/a.wav  b.wav        <- 同一说话人的不同录音
      spk02/c.wav
    anon_dir/
      anon_a.wav  anon_b.wav    <- 对应原始音频的匿名版本

样本对：
  - 正对(同一个人)：同一 spk 目录内两两配对  -> 应高相似
  - 负对(不同人/匿名)：原始 vs 其匿名版本    -> 应低相似
在相似度阈值上同时最小化 FAR 与 FRR 的交点即为 EER。

若目录没有按说话人分组（无法构造正对），自动降级为
"匿名前后相似度统计"模式，作为匿名强度的参考指标。

用法：
    python -m eval.eer --orig_dir data/orig --anon_dir outputs
"""
import argparse
import os

import numpy as np
from speechbrain.inference.speaker import SpeakerRecognition


def cosine_sim(e1: np.ndarray, e2: np.ndarray) -> float:
    return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_dir", required=True, help="按说话人子目录组织的原始音频")
    ap.add_argument("--anon_dir", required=True, help="匿名音频目录")
    args = ap.parse_args()

    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ecapa = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
    )

    # ---- 收集嵌入 ----
    # speechbrain 新版 encode_batch 只接受 torch.Tensor；用 librosa 读成 16k 单声道再转 tensor
    import librosa
    import torch

    emb_cache = {}  # 绝对路径 -> 嵌入向量（懒计算，避免重复编码）
    def embed(path):
        if path not in emb_cache:
            sig, _ = librosa.load(path, sr=16000, mono=True)
            sig_t = torch.from_numpy(sig)
            emb_cache[path] = ecapa.encode_batch(sig_t[None, :]).reshape(-1).cpu().numpy()
        return emb_cache[path]

    # 正对：同说话人目录内两两配对
    pos_pairs, neg_pairs = [], []
    for spk in sorted(os.listdir(args.orig_dir)):
        spk_dir = os.path.join(args.orig_dir, spk)
        if not os.path.isdir(spk_dir):
            continue
        wavs = sorted(f for f in os.listdir(spk_dir) if f.endswith(".wav"))
        for i in range(len(wavs)):
            for j in range(i + 1, len(wavs)):
                pos_pairs.append((os.path.join(spk_dir, wavs[i]), os.path.join(spk_dir, wavs[j])))
    # 负对：原始 vs 匿名
    for fn in sorted(os.listdir(args.anon_dir)):
        if not fn.endswith(".wav"):
            continue
        stem = fn[len("anon_"):] if fn.startswith("anon_") else fn
        for spk in os.listdir(args.orig_dir):
            cand = os.path.join(args.orig_dir, spk, stem)
            if os.path.isfile(cand):
                neg_pairs.append((cand, os.path.join(args.anon_dir, fn)))
                break

    print(f"正对: {len(pos_pairs)}  负对: {len(neg_pairs)}")

    if not pos_pairs:
        # ---- 降级模式：只统计匿名前后相似度（单说话人场景）----
        sims = []
        for orig, anon in neg_pairs:
            sims.append(cosine_sim(embed(orig), embed(anon)))
        if not sims:
            print("无样本"); return
        print(f"[降级模式] 原始↔匿名平均相似度: {np.mean(sims):.3f} "
              f"(越低=匿名越强；构造正对需按说话人分组，才能算 EER)")
        return

    # ---- 完整 EER ----
    scores, labels = [], []
    for a, b in pos_pairs:
        scores.append(cosine_sim(embed(a), embed(b)))
        labels.append(1)
    for a, b in neg_pairs:
        scores.append(cosine_sim(embed(a), embed(b)))
        labels.append(0)
    scores = np.array(scores)
    labels = np.array(labels)

    # 在阈值网格上扫描，找 FAR 与 FRR 差距最小的点作为 EER
    best_eer, best_th = 1.0, 0.5
    for th in np.linspace(0.0, 1.0, 2001):
        pred = (scores >= th).astype(int)
        far = np.mean((pred == 1) & (labels == 0))  # 误接受：不同人被当成同一人
        frr = np.mean((pred == 0) & (labels == 1))  # 误拒绝：同一人被当成不同人
        diff = abs(far - frr)
        if diff < abs(best_eer - (far + frr) / 2) or best_eer == 1.0:
            best_eer = (far + frr) / 2
            best_th = th

    print(f"EER: {best_eer * 100:.2f}%  (阈值={best_th:.3f})")
    print(f"正对相似度均值: {np.mean(scores[labels == 1]):.3f}  负对相似度均值: {np.mean(scores[labels == 0]):.3f}")


if __name__ == "__main__":
    main()
