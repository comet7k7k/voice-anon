# eval 评测脚本

论文三个核心指标，脚本可直接运行：

| 脚本 | 指标 | 含义 | 方向 |
|---|---|---|---|
| `wer.py` | WER | 匿名后文本保留度 | 越低越好 |
| `eer.py` | EER | 匿名强度（说话人难识别度） | 越高越好 |
| `uar.py` | UAR | 匿名后情感保留度 | 越高越好 |

## 数据准备（RAVDESS，约 208MB）

官方：https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip
（语音部分一个 zip，md5: bc696df654c87fed845eb13823edef8a，解压得 Actor_01~24 目录）

```bash
# 1. 组织评测数据：按说话人分组 + 生成真实文本标注
python scripts/setup_eval_data.py --ravdess_dir D:/RAVDESS --actors 01,02,03,04 --max_per_actor 8
# 2. 批量匿名化（输出 outputs/anon_*.wav）
python scripts/batch_anonymize.py --orig_dir data/orig --out_dir outputs
# 3. 跑评测
python -m eval.wer --orig_dir data/orig --anon_dir outputs --ref_txt data/ref_txt
python -m eval.eer --orig_dir data/orig --anon_dir outputs
python -m eval.uar --orig_dir data/orig --anon_dir outputs
```

## 数据组织（重要）

```
data/orig/                 # 原始音频（按说话人建子目录，eer 需要）
  spk01/001.wav  002.wav   # 同一说话人的不同录音
  spk02/003.wav
outputs/                   # 匿名音频（pipeline 输出）
  anon_001.wav  anon_002.wav  anon_003.wav
data/ref_txt/              # 可选：真实文本标注（wer 更严谨）
  001.txt  002.txt  003.txt
```

## 运行

```bash
python -m eval.wer --orig_dir data/orig --anon_dir outputs [--ref_txt data/ref_txt]
python -m eval.eer --orig_dir data/orig --anon_dir outputs
python -m eval.uar --orig_dir data/orig --anon_dir outputs
```

## 说明与局限

- **eer.py**：正对=同说话人两段不同录音，负对=原始↔匿名；数据未按说话人分组时自动降级为"匿名前后相似度统计"。
- **uar.py**：用 transformers 加载 SUPERB `wav2vec2-base-superb-er`（IEMOCAP 4 类，首次运行走 HF 镜像下载 ~240MB）；真实标签优先解析 RAVDESS 文件名情感码，非 RAVDESS 命名时回退为原始音频预测（近似）。
- **命名约定**：匿名文件 `anon_<原名>` 与原始文件 `<原名>` 配对。
