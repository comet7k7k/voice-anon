"""整条匿名化流水线：wav → Whisper 文本 + F0/情感原型注意力 → VITS 匿名音频。"""
import os
import numpy as np
import torch

from .whisper_model import WhisperSTT
from .vits import VITSLJSpeech
from .emotion import extract_f0, build_emotion_prototypes, PrototypeAttention
from .film import FiLM

# 情感类顺序，必须与 scripts/compute_emotion_prototypes.py 的 PROTO_ORDER 一致
PROTO_ORDER = ["neu", "hap", "sad", "ang"]
DEFAULT_EMOTION_MEANS = {"neu": 180.0, "hap": 220.0, "sad": 140.0, "ang": 200.0}
DEFAULT_PROTOTYPES_PATH = "data/emotion_prototypes.npy"


class AnonymizationPipeline:
    def __init__(self, out_dir: str = "outputs", prototypes_path: str = DEFAULT_PROTOTYPES_PATH,
                 use_prototypes: bool = True):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir

        # 情感原型：优先加载 RAVDESS 计算出的真实质心；没有则用默认占位值。
        # use_prototypes=False 用于消融（Table 4 对照）：情感原型置零，注意力仅依赖随机 Wp/Wf。
        if use_prototypes:
            self.prototypes, self.prototype_source = self._load_prototypes(prototypes_path)
        else:
            self.prototypes = np.zeros(len(PROTO_ORDER), dtype=np.float32)
            self.prototype_source = "OFF (ablation: no emotion prototype)"
        print(f"[pipeline] 情感原型来源: {self.prototype_source} -> {self.prototypes}")

        self.attn = PrototypeAttention(n_proto=len(self.prototypes))
        # feat_dim 取原型维度，用于 FiLM（论文模块，见 README 局限）
        self.film = FiLM(feat_dim=len(self.prototypes))
        # 重模型懒加载，避免 import 时就下载
        self._stt = None
        self._vits = None

    @staticmethod
    def _load_prototypes(prototypes_path: str):
        """返回 (prototypes数组, 来源描述)。真实 npy 优先，否则默认占位值。"""
        if os.path.isfile(prototypes_path):
            arr = np.load(prototypes_path).astype(np.float32).reshape(-1)
            if arr.size == len(PROTO_ORDER):
                return arr, f"real (from {prototypes_path})"
            print(f"[warn] {prototypes_path} 维度不符(期望{len(PROTO_ORDER)})，回退默认值")
        return build_emotion_prototypes(DEFAULT_EMOTION_MEANS), "default placeholder (TODO: 跑 RAVDESS 脚本替换)"

    @property
    def stt(self):
        if self._stt is None:
            self._stt = WhisperSTT("small.en")
        return self._stt

    @property
    def vits(self):
        if self._vits is None:
            self._vits = VITSLJSpeech()
        return self._vits

    def run(self, in_wav: str, out_name: str = "anon.wav") -> dict:
        # 1. 文本
        text = self.stt.transcribe(in_wav)
        # 2. F0 + 情感原型注意力（论文模块，产出可视化分析结果）
        f0 = extract_f0(in_wav)
        f0_t = torch.tensor(f0, dtype=torch.float32).unsqueeze(-1)
        proto_t = torch.tensor(self.prototypes, dtype=torch.float32)
        f_integrated, attn = self.attn(f0_t, proto_t)
        # 3. FiLM（论文模块；标准 VITS 文本嵌入不对外暴露，见 README 局限）
        # film_out = self.film(text_embed, f_integrated)
        # 4. VITS 合成固定伪说话人音色（匿名主干）
        out_wav = os.path.join(self.out_dir, out_name)
        self.vits.synthesize(text, out_wav)
        return {
            "text": text,
            "out_wav": out_wav,
            "f0_mean": float(np.mean(f0[f0 > 0])) if np.any(f0 > 0) else 0.0,
            "n_frames": int(len(f0)),
            "attn_shape": list(attn.shape),
            "prototype_source": self.prototype_source,
        }
