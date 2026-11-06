"""TTS：VITS，LJ Speech 单说话人预训练权重。

论文使用 jaywalnut310/vits 在 LJ Speech 上的预训练模型。
这里用 Coqui TTS 内置的 `ljspeech/vits`（与官方同源）。
模型权重已手动下载解压到 Coqui 缓存目录：
    %LOCALAPPDATA%\\tts\\tts_models--en--ljspeech--vits\\
（含 config.json + model_file.pth），加载时不再联网下载。
"""


class VITSLJSpeech:
    def __init__(self, model_name: str = "tts_models/en/ljspeech/vits"):
        from TTS.api import TTS
        import torch
        self.tts = TTS(model_name)
        # 显式放到 GPU（Coqui 新 API 建议 tts.to(device)，gpu= 参数已标记废弃）
        self.tts.to("cuda" if torch.cuda.is_available() else "cpu")

    def synthesize(self, text: str, out_path: str) -> str:
        self.tts.tts_to_file(text=text, file_path=out_path)
        return out_path
