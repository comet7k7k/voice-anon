"""STT：OpenAI Whisper-small.en 文本提取。"""
import librosa
import numpy as np


class WhisperSTT:
    def __init__(self, model_name: str = "small.en", device: str = None):
        import whisper  # openai-whisper
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = whisper.load_model(model_name, device=self.device)

    def transcribe(self, wav_path: str) -> str:
        # 用 librosa 直接读 wav 为 numpy 数组（16kHz 单声道），
        # 这样 whisper.transcribe 就不会再去调用外部 ffmpeg 进程（Windows 上常缺失，会报 WinError 2）
        audio, _ = librosa.load(wav_path, sr=16000, mono=True)
        result = self.model.transcribe(audio)
        return (result.get("text") or "").strip()
