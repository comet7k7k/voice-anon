"""批量匿名化：把 data/orig/spkXX/ 下所有 wav 跑一遍流水线，输出到 outputs/。

输出命名 anon_<原名>.wav，与 eval/wer.py、eval/eer.py 的配对约定一致。

用法：
    python scripts/batch_anonymize.py --orig_dir data/orig --out_dir outputs
"""
import argparse
import os
import sys
import time

# 把项目根目录加入 sys.path，直接复用 app 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.pipeline import AnonymizationPipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--orig_dir", default="data/orig", help="原始音频根目录（递归扫描 wav）")
    ap.add_argument("--out_dir", default="outputs", help="匿名音频输出目录")
    ap.add_argument("--no_prototype", action="store_true",
                    help="消融：不加载情感原型（原型置零），输出到 out_dir 下 outputs_no_proto 类似独立目录")
    args = ap.parse_args()

    pipe = AnonymizationPipeline(out_dir=args.out_dir, use_prototypes=not args.no_prototype)

    wavs = []
    for root, _, files in os.walk(args.orig_dir):
        for fn in sorted(files):
            if fn.endswith(".wav"):
                wavs.append(os.path.join(root, fn))
    if not wavs:
        print(f"在 {args.orig_dir} 下没找到 wav，先跑 scripts/setup_eval_data.py 准备数据")
        return

    print(f"共 {len(wavs)} 个音频，开始匿名化（CPU 上每个约 5~15 秒，耐心等）...")
    t0 = time.time()
    for i, path in enumerate(wavs, 1):
        out_name = "anon_" + os.path.basename(path)
        try:
            res = pipe.run(path, out_name=out_name)
            print(f"[{i}/{len(wavs)}] {os.path.basename(path)} -> {out_name} "
                  f"(f0_mean={res['f0_mean']:.1f}Hz, {res['n_frames']}帧)")
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(wavs)}] {os.path.basename(path)} 失败: {e}")
    print(f"完成，耗时 {time.time() - t0:.0f}s，输出目录: {args.out_dir}")


if __name__ == "__main__":
    main()
