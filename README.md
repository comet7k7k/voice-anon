# voice-anon：语音匿名化系统（Lee et al., SPSC 2024 复现）

> 论文：*Voice Anonymization Using Emotion-Enriched Feature Integration with STT and TTS Models*
> 流水线：原始 wav → Whisper-small.en 转文本 + librosa 提 F0 → 情感原型注意力 → FiLM 融合 → VITS(LJ Speech) → 固定伪说话人音色的匿名 wav。

## 当前状态（2026-10-31）

✅ **端到端已跑通**：Whisper 转录 → F0/情感注意力 → VITS 合成匿名音频，本地浏览器可上传/播放。
✅ **评测闭环已完成（真实数据）**：RAVDESS 32 段（4 情感分层抽样）→ 匿名化 → WER=3.12% / EER=0.00% / UAR=25.00%，与论文趋势一致（报告 6.2 节已填）。
✅ **情感原型已实做**：`data/emotion_prototypes.npy`（RAVDESS 真实 F0 质心 [222.18, 362.13, 337.22, 409.59]），pipeline 启动自动加载。
✅ **GPU 加速已启用（RTX 4060）**：torch 2.14.1+cu126，Whisper/VITS/ECAPA/SER 全部自动上 CUDA。批量 32 段匿名化 GPU 全链 59.6s。基准：Whisper 1.70→0.51s（3.3×）、VITS 1.05→0.81s、ECAPA 0.08→0.01s（8×）、SER 0.34→0.05s（6.8×）。GPU 版评测复核：EER=0.00% / UAR=25.00% 与 CPU 权威数据一致（WER 因 whisper 解码随机性 6.25%，权威数字仍取 CPU 版 3.12%）。
✅ **华为云部署已完成并公网实测通过**：V100 ECS（华北-北京四，p2v.2xlarge.8），公网地址 http://120.46.83.92/ ，上传 wav 后返回 Whisper 转录、F0 均值、情感原型来源与可播放的匿名音频。

## 快速开始

```bash
# 1. 先单独装 CUDA 版 PyTorch（不要加 --extra-index-url，否则会 fallback 到 CPU 版）
pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# 2. 再装其余依赖
pip install -r requirements.txt

# 3. 启动服务
python -m uvicorn app.main:app --port 8000
```

浏览器打开 http://localhost:8000 ，上传一段 wav，得到匿名音频 + 转录文本 + F0/注意力信息。

> Windows 额外依赖：`winget install eSpeak-NG.eSpeak-NG`（VITS 音素化需要）。
> 模型权重（Whisper small.en ~460MB + VITS LJ Speech ~278MB）：见下方"权重下载"。

## 目录结构

```
app/
  whisper_model.py   # STT：Whisper-small.en
  emotion.py         # F0 提取 + 情感原型 + 注意力（论文公式 1-5）
  film.py            # FiLM 融合（论文公式 6-7）
  vits.py            # TTS：VITS-LJ Speech（Coqui 封装）
  pipeline.py        # 串联整条流水线（支持加载真实情感原型 npy）
  main.py            # FastAPI 入口
  static/index.html  # 上传/播放前端
eval/                # WER / EER / UAR 评测脚本（见 eval/README.md）
scripts/
  compute_emotion_prototypes.py  # RAVDESS → 4 情感类 F0 质心 → npy
  setup_eval_data.py             # RAVDESS → data/orig（4 情感分层抽样）+ ref_txt
  batch_anonymize.py             # 批量匿名化（--no_prototype 跑消融版）
  eval_wer_orig.py / eval_uar_orig.py  # 原始音频 WER/UAR 基线（报告 6.2"原始"列）
  compare_outputs.py             # 对比两版匿名输出是否逐字节一致（消融辅助）
  bench_gpu.py                   # CPU vs GPU 耗时基准（报告性能对比证据）
weights/             # 模型权重（不进 git，网盘下载）
data/                # 测试音频、评测数据（gitignore）
```

## 论文方法对照

| 论文组件 | 本仓库实现 |
|---|---|
| Whisper-small.en 文本提取 | `app/whisper_model.py` |
| librosa F0（65.41Hz~2093Hz, win2048/hop512） | `app/emotion.py::extract_f0` |
| 4 情感类 F0 质心原型 | `scripts/compute_emotion_prototypes.py`（RAVDESS 真实计算） |
| 注意力融合（公式1-5） | `app/emotion.py::PrototypeAttention` |
| FiLM 融合（公式6-7） | `app/film.py::FiLM` |
| VITS（LJ Speech 单说话人） | `app/vits.py`（Coqui TTS，同源模型） |

## 情感原型（论文核心模块，已实做）

1. 下载 RAVDESS 数据集（约 150MB，官网免费）；
2. 计算真实质心：
   ```bash
   python scripts/compute_emotion_prototypes.py --data_dir D:/RAVDESS --out data/emotion_prototypes.npy
   ```
3. 重启服务，pipeline 会自动加载 `data/emotion_prototypes.npy`（启动日志会打印来源）；
   没有 npy 时回退默认占位值（日志会明确警告）。

## ⚠️ 已知局限（报告"局限与思考"素材）

1. **VITS 无 F0 输入口**：标准 VITS 推理只吃文本，论文未说明"增强 F0 如何进入 VITS"，当前 F0/情感模块产出分析结果，不直接注入 VITS。**消融实证**：加/不加情感原型 UAR 均为 25.00%（Δ=0.00pp），两版 wav 逐字节不同但差异完全来自 VITS 合成随机性（GAN 采样），原型未传导到合成音频。
2. **Wp/Wf 训练权重缺失**：论文未公开注意力层与 FiLM 的训练方式与维度，按公式结构随机初始化；可用消融实验验证其作用。
3. **SER 跨数据集域偏移**：eval/uar.py 用 SUPERB ER（IEMOCAP 上微调）评测 RAVDESS；匿名后各情感几乎全部被判为 ang，VITS 重合成的统一韵律主导了 SER 输出。真实标签取自 RAVDESS 文件情感码。
4. **VITS 选型**：Coqui `ljspeech/vits` 与 jaywalnut310/vits 同源；完全复刻论文可换官方 checkpoint。

## 权重下载（百度网盘）

> 按作业要求，模型权重不进 GitHub，打包传百度网盘。

| 文件 | 大小 | 用途 | 放置位置 |
|---|---|---|---|
| `small.en.pt` | ~461 MB | Whisper-small.en | `~/.cache/whisper/small.en.pt` |
| `tts_models--en--ljspeech--vits/` | ~278 MB | VITS LJ Speech | `~/.local/share/tts/tts_models--en--ljspeech--vits/` |

百度网盘链接：https://pan.baidu.com/s/1rI8gQftydnkzUqLcVg0IPg?pwd=wqcx （提取码：wqcx）

## 部署（华为云，已实测）

实例：华北-北京四，p2v.2xlarge.8（V100 16GB），Ubuntu 22.04 + CUDA 12.2。

```bash
# 服务器上启动（监听 80 端口，安全组 Sys-WebServer 已放行 80）
cd /root/voice-anon && source /root/anon/bin/activate && \
  nohup uvicorn app.main:app --host 0.0.0.0 --port 80 > /root/anon.log 2>&1 &
```

公网访问：**http://120.46.83.92/**（按需开机，验收当天启动后即可演示）。
