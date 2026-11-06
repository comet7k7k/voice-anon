"""把 RAVDESS 组织成评测数据：data/orig/spkXX/ + data/ref_txt/。

RAVDESS 自带说话人分组（Actor_01~24 = 24 个说话人，奇男偶女），
且每段音频只对应两句固定文本（statement 01/02），天然满足：
  - eval/eer.py 需要"按说话人子目录"组织（正对=同演员两段，负对=原始↔匿名）
  - eval/wer.py 需要真实文本标注（ref_txt，比 Whisper 转写更严谨）

用法（先下载并解压 Audio_Speech_Actors_01-24.zip）：
    python scripts/setup_eval_data.py --ravdess_dir D:/RAVDESS
    # 可选：--actors 01,02,03,04 --max_per_actor 8 控制规模（CPU 上跑得快）
"""
import argparse
import os
import re
import shutil
from collections import defaultdict

# RAVDESS statement 段 → 固定文本（官方定义）
STATEMENT_TEXTS = {
    "01": "Kids are talking by the door.",
    "02": "Dogs are sitting by the door.",
}


def parse_filename(fn: str):
    """解析 RAVDESS 文件名，返回 (emotion_code, statement_code, actor_code) 或 None。"""
    m = re.match(r"(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)", fn)
    if not m:
        return None
    modality, channel, emotion, _, statement, _, actor = m.groups()
    # 只要音频(03)-语音(01)
    if modality != "03" or channel != "01":
        return None
    return emotion, statement, actor


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ravdess_dir", required=True, help="RAVDESS 解压目录（含 Actor_01~24 子目录）")
    ap.add_argument("--out_dir", default="data", help="输出根目录（默认 data/）")
    ap.add_argument("--actors", default=None,
                    help="使用的演员列表，逗号分隔，如 01,02,03,04（默认全部 24 个）")
    ap.add_argument("--max_per_actor", type=int, default=8,
                    help="每个演员最多取几个文件（控制评测规模，默认 8）")
    args = ap.parse_args()

    actors = args.actors.split(",") if args.actors else None
    orig_dir = os.path.join(args.out_dir, "orig")
    ref_dir = os.path.join(args.out_dir, "ref_txt")
    os.makedirs(orig_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)

    n_copied = n_skipped = 0
    for d in sorted(os.listdir(args.ravdess_dir)):
        if not re.fullmatch(r"Actor_\d{2}", d):
            continue
        actor = d[-2:]
        if actors and actor not in actors:
            continue
        actor_dir = os.path.join(args.ravdess_dir, d)
        wavs = sorted(f for f in os.listdir(actor_dir) if f.endswith(".wav"))
        # 按情感码分层：论文 4 类（01/03/04/05）各取若干，保证评测覆盖全部情感
        by_emotion: dict[str, list[str]] = defaultdict(list)
        for fn in wavs:
            parsed = parse_filename(fn)
            if parsed is not None:
                by_emotion[parsed[0]].append(fn)
        per_class = max(1, args.max_per_actor // 4)
        chosen: list[str] = []
        for emo in ["01", "03", "04", "05"]:
            chosen.extend(by_emotion.get(emo, [])[:per_class])

        for fn in chosen:
            parsed = parse_filename(fn)
            if parsed is None:
                n_skipped += 1
                continue
            _, statement, _ = parsed
            # 说话人目录：spkXX
            spk_dir = os.path.join(orig_dir, f"spk{actor}")
            os.makedirs(spk_dir, exist_ok=True)
            shutil.copy2(os.path.join(actor_dir, fn), os.path.join(spk_dir, fn))
            # 真实文本标注（与原始 wav 同名 .txt）
            with open(os.path.join(ref_dir, fn.rsplit(".", 1)[0] + ".txt"),
                      "w", encoding="utf-8") as f:
                f.write(STATEMENT_TEXTS.get(statement, ""))
            n_copied += 1

    print(f"已复制 {n_copied} 个音频到 {orig_dir}/spkXX/，标注写入 {ref_dir}/")
    if n_skipped:
        print(f"跳过 {n_skipped} 个非目标文件")
    print("\n下一步：\n"
          f"  1) python scripts/batch_anonymize.py --orig_dir {orig_dir}\n"
          "  2) python -m eval.wer --orig_dir data/orig --anon_dir outputs --ref_txt data/ref_txt\n"
          "  3) python -m eval.eer --orig_dir data/orig --anon_dir outputs")


if __name__ == "__main__":
    main()
