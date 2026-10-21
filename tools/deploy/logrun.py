#!/usr/bin/env python3
"""启动并守护一个子进程，把它的 stdout/stderr 写进**带大小上限**的日志。

用法：
    logrun.py --log run/serve_long.log --max-mb 32 --keep 3 -- \
        env FOO=1 ./bin/lingxi-bench serve ...

行为：
- 子进程输出按行实时落盘（不做全缓冲）；
- 文件超过 max-mb 就轮转：xxx.log → xxx.log.1 → xxx.log.2 …，只留 keep 份；
- 收到 SIGTERM/SIGINT 转发给子进程，等它退出后自己再退出；
- pid 文件里写的是**本进程 pid**（down.sh 杀它即可连带子进程）。

为什么不用 shell 重定向：`cmd > log 2>&1 &` 的日志会无限增长；
按 pid 杀时也只杀到管道的一端，长跑服务不适合。
"""

import argparse
import os
import select
import signal
import subprocess
import sys


def rotate(path: str, keep: int) -> None:
    for i in range(keep - 1, 0, -1):
        src = f"{path}.{i}"
        if os.path.exists(src):
            os.replace(src, f"{path}.{i + 1}")
    os.replace(path, f"{path}.1")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--max-mb", type=int, default=32)
    ap.add_argument("--keep", type=int, default=3)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()

    cmd = a.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("logrun: 缺少要执行的命令", file=sys.stderr)
        return 2

    os.makedirs(os.path.dirname(a.log) or ".", exist_ok=True)
    max_bytes = max(1, a.max_mb) * 1024 * 1024

    child = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        start_new_session=False,
    )

    def forward(sig, _frame):
        try:
            child.send_signal(sig)
        except ProcessLookupError:
            pass

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)

    f = open(a.log, "ab", buffering=0)
    written = os.path.getsize(a.log)
    assert child.stdout is not None
    while True:
        # 用 select 带超时：子进程退出后即使管道还被某个 fd 持着，也不会把 wrapper 卡死
        readable, _, _ = select.select([child.stdout], [], [], 0.5)
        if readable:
            line = child.stdout.readline()
            if not line:
                break
            f.write(line)
            written += len(line)
            if written >= max_bytes:
                f.close()
                rotate(a.log, a.keep)
                f = open(a.log, "ab", buffering=0)
                written = 0
        elif child.poll() is not None:
            # 子进程已经退出：把管道里剩下的读干（非阻塞）后收尸
            while True:
                readable, _, _ = select.select([child.stdout], [], [], 0)
                if not readable:
                    break
                line = child.stdout.readline()
                if not line:
                    break
                f.write(line)
            break

    f.close()
    rc = child.wait()
    return rc if rc >= 0 else 128 - rc


if __name__ == "__main__":
    sys.exit(main())
