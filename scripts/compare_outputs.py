"""对比两版匿名输出是否一致（消融验证辅助）：有原型 outputs/ vs 无原型 outputs_no_proto/。

用途：验证"情感原型是否传导到合成音频"。若 VITS 仅依赖文本，两版 wav 应逐字节相同
（或仅存在 TTS 合成随机性差异）。
"""
import hashlib
import os
import sys


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    dir_a = sys.argv[1] if len(sys.argv) > 1 else "outputs"
    dir_b = sys.argv[2] if len(sys.argv) > 2 else "outputs_no_proto"
    files_a = set(f for f in os.listdir(dir_a) if f.endswith(".wav"))
    files_b = set(f for f in os.listdir(dir_b) if f.endswith(".wav"))
    common = sorted(files_a & files_b)
    identical = 0
    different = []
    for fn in common:
        if md5(os.path.join(dir_a, fn)) == md5(os.path.join(dir_b, fn)):
            identical += 1
        else:
            different.append(fn)
    print(f"{dir_a} 与 {dir_b} 共配对 {len(common)} 个，逐字节相同 {identical} 个，不同 {len(different)} 个")
    if different:
        print("不同文件（前 10 个）:", different[:10])


if __name__ == "__main__":
    main()
