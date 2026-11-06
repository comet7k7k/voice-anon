"""FiLM 融合层（Lee et al., SPSC 2024，公式 6-7）。"""
import torch.nn as nn


class FiLM(nn.Module):
    """
        gamma = 1 + Linear(Normalize(F_integrated))
        FiLM(T | gamma) = gamma ⊙ T   （⊙ 逐元素乘）
    论文用 gamma 调制文本 embedding。
    注意：标准 VITS 推理时文本 embedding 不对外暴露，本层按论文公式实现，
    TODO: 若接入可观测中间特征的 TTS（如 VITS 内部 encoder），在此替换。
    """

    def __init__(self, feat_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(feat_dim)
        self.linear = nn.Linear(feat_dim, feat_dim)

    def forward(self, text_embed, f_integrated):
        gamma = 1.0 + self.linear(self.norm(f_integrated))
        return gamma * text_embed
