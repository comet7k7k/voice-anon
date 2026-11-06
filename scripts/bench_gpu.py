"""CPU vs GPU 耗时基准（报告性能对比证据）。

对同一输入分别测量：
  1. Whisper STT 转写（small.en，CPU / CUDA）
  2. VITS 合成（Coqui LJ Speech，CPU / CUDA）
  3. ECAPA-TDNN 说话人嵌入（speechbrain，CPU / CUDA）
  4. wav2vec2 SER 情感推理（SUPERB ER，CPU / CUDA）

用法（在 voice-anon 根目录）：
    python scripts/bench_gpu.py --wav data/test_en.wav
输出各设备耗时秒数（多次取最小，减少抖动）。

说明：GPU 推理在相同环境内通过 device 参数切换测量，CPU 模式不要求 CPU 版 torch。
"""
import argparse
import time

import librosa
import numpy as np
import torch


def bench(label: str, fn, repeat: int = 2) -> float:
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    print(f"  {label:28s} {best:.2f}s")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="测试音频（英文 wav）")
    ap.add_argument("--text", default="Hello, this is a test recording for GPU benchmark.")
    ap.add_argument("--out", default="data/bench_out.wav", help="VITS 合成输出路径")
    args = ap.parse_args()

    print("== 1. Whisper STT（small.en）==")
    import whisper
    audio, _ = librosa.load(args.wav, sr=16000, mono=True)
    for dev in ["cpu", "cuda"]:
        if dev == "cuda" and not torch.cuda.is_available():
            continue
        m = whisper.load_model("small.en", device=dev)
        fp16 = dev == "cuda"
        bench(f"whisper_{dev}", lambda: m.transcribe(audio, fp16=fp16))

    print("== 2. VITS 合成（Coqui ljspeech）==")
    from TTS.api import TTS
    for dev in ["cpu", "cuda"]:
        if dev == "cuda" and not torch.cuda.is_available():
            continue
        tts = TTS("tts_models/en/ljspeech/vits", gpu=(dev == "cuda"))
        bench(f"vits_{dev}", lambda: tts.tts_to_file(text=args.text, file_path=args.out))

    print("== 3. ECAPA-TDNN 说话人嵌入（speechbrain）==")
    from speechbrain.inference.speaker import SpeakerRecognition
    for dev in ["cpu", "cuda"]:
        if dev == "cuda" and not torch.cuda.is_available():
            continue
        ecapa = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": dev},
        )
        sig, _ = librosa.load(args.wav, sr=16000, mono=True)
        sig_t = torch.from_numpy(sig)
        bench(f"ecapa_{dev}", lambda: ecapa.encode_batch(sig_t[None, :]).reshape(-1).cpu().numpy())

    print("== 4. wav2vec2 SER 情感推理（SUPERB ER）==")
    from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
    ser = "superb/wav2vec2-base-superb-er"
    model = Wav2Vec2ForSequenceClassification.from_pretrained(ser)
    fe = Wav2Vec2FeatureExtractor.from_pretrained(ser)
    inputs = fe(audio, sampling_rate=16000, return_tensors="pt")
    for dev in ["cpu", "cuda"]:
        if dev == "cuda" and not torch.cuda.is_available():
            continue
        mm = model.to(dev)
        inp = {k: v.to(dev) for k, v in inputs.items()}
        with torch.no_grad():
            bench(f"ser_{dev}", lambda: mm(**inp))


if __name__ == "__main__":
    main()
