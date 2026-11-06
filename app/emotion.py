"""F0 提取 + 情感原型 + 注意力融合（Lee et al., SPSC 2024，公式 1-5）。"""
import numpy as np
import librosa
import torch
import torch.nn as nn

# 论文参数：fmin=65.41Hz(C2), fmax=2093Hz(C7), win=2048, hop=512
FMIN = 65.41
FMAX = 2093.0
SR = 22050
FRAME_LEN = 2048
HOP = 512


def extract_f0(wav_path: str, sr: int = SR) -> np.ndarray:
    """用 librosa.pyin 提取 F0 曲线，未发声帧补 0。返回 shape=(n_frames,)。"""
    y, _ = librosa.load(wav_path, sr=sr)
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=FMIN, fmax=FMAX, sr=sr, frame_length=FRAME_LEN, hop_length=HOP
    )
    f0 = np.nan_to_num(f0, nan=0.0)
    return f0.astype(np.float32)


def build_emotion_prototypes(emotion_f0_means: dict) -> np.ndarray:
    """论文：从 IEMOCAP 训练集按 4 个情感类求 F0 质心作为 prototype。

    emotion_f0_means: {情感类名: 该类 F0 均值}。
    TODO: 用 RAVDESS 音频批处理计算，替换这里的默认值。
    """
    return np.array(list(emotion_f0_means.values()), dtype=np.float32)


class PrototypeAttention(nn.Module):
    """公式 1-5：
        p = ReLU(Wp · Prototype)
        f = ReLU(Wf · F0)
        Similarity = p^T f ; AttentionWeights = softmax(Similarity)
        F_integrated = AttentionWeights × f
    注意：论文未公开 Wp/Wf 维度与训练权重，此处按公式结构初始化；
    TODO: 用 IEMOCAP/RAVDESS 训练集拟合 Wp/Wf。
    """

    def __init__(self, n_proto: int = 4):
        super().__init__()
        self.Wp = nn.Linear(n_proto, n_proto)
        # f0 每帧作为 1 维标量特征，用线性层升到 n_proto 维便于对齐
        self.Wf = nn.Linear(1, n_proto)

    def forward(self, f0: torch.Tensor, prototypes: torch.Tensor):
        """
        f0: (n_frames, 1)；prototypes: (n_proto,)
        返回 F_integrated (n_proto, 1)，attn (n_proto, n_frames)。
        """
        p = torch.relu(self.Wp(prototypes))                 # (n_proto,)
        f = torch.relu(self.Wf(f0))                          # (n_frames, n_proto)
        sim = p.unsqueeze(0) * f                             # (n_frames, n_proto)
        attn = torch.softmax(sim, dim=1)                     # 跨情感类归一
        f_integrated = torch.matmul(attn.transpose(0, 1), f0)  # (n_proto, 1)
        return f_integrated, attn
