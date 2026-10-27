#!/usr/bin/env python3
"""抓"一轮软卡 3.3 s"的现场。

**软卡是什么**：`spec_bench.sh` 偶尔（约 1/6 跑）会出现某一轮 3.3 s（其余轮 58 ms），
把 200 token 的平均吞吐从 96 打到 37。`spec_bench.sh` 已按 `max > 3×p50` 作废这种跑，
但根因一直没钉住。这个脚本同时排除四个候选，**只在异常时打一行**：

  1. 时钟跳变     → CLOCK_MONOTONIC（Rust `Instant` 就是它）vs CLOCK_MONOTONIC_RAW
                     （raw 不受 NTP/timekeeping 校正；两者差突然变 ⇒ 计时假象）
  2. 内存回收     → `/proc/pressure/memory` 的 total + vmstat 的 pgscan/allocstall
  3. 进程在睡觉   → lingxi-bench 的 Δcpu / Δwall（Δcpu ≪ Δwall ⇒ 被阻塞在系统调用/驱动里）
  4. D 态线程数   → /proc/<pid>/task/*/stat 里 state == 'D' 的个数

用法（**先起它，再跑 spec_bench**；跑完 Ctrl-C 看 /tmp/soft_stall.log）：

    setsid nohup python3 -u tools/golden/soft_stall_watch.py > /tmp/soft_stall.log 2>&1 &
    bash tools/golden/spec_bench.sh s1 200 1

判读：
* `mono+3.3 raw+0.25` ⇒ **时钟跳变**，软卡是假象（吞吐被低估，实际墙钟没变）；
* `vmoved=True` 或 PSI total 涨 ⇒ 内存回收；
* `wall+3.30 cpu+0.02 gap+3.28` ⇒ 进程**真的被冻住**，且没在算 ⇒ 下一枪是
  `strace -f -T -p <pid>` 或 `cat /proc/<pid>/task/*/stack` 抓它在哪个系统调用里；
* `wall+3.30 cpu+3.25 gap+0.05` ⇒ 进程在**满核空转**（比如自旋/大循环），CPU 侧问题。

⚠️ 开销：每 tick 扫一遍 /proc（含每进程的线程），约占单核 5%；它自己可能让结果偏慢 1%，
所以**只用于诊断，不要和吞吐跑的判据混在一起**。
已排除的候选（2026-10-25，都是本脚本量的）：
  * 内存回收 —— 633 点 / 645 s 采样，PSI total 与三个 pgscan 计数**一位没动**；
  * 时钟跳变 —— 同一窗口 `drift(mono-raw)` 只从 +23.3610 缓慢漂到 +23.3665，**无跳变**。
"""
import os
import time

M, R = time.CLOCK_MONOTONIC, time.CLOCK_MONOTONIC_RAW
TICK = 0.25            # 采样周期（秒）
WFULL = 0.35           # 单步墙钟超过它就报（正常 ~TICK）
CPU_GAP = 0.12         # Δwall − Δcpu 超过它 ⇒ 这段墙钟里进程没在算
HZ = os.sysconf("SC_CLK_TCK")
TARGET = os.environ.get("SOFT_STALL_PROC", "lingxi-bench")


def snapshot():
    """{pid: (累计 CPU 秒, D 态线程数)}，只看 TARGET 进程。"""
    out = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/comm") as f:
                if f.read().strip() != TARGET:
                    continue
            with open(f"/proc/{pid}/stat") as f:
                st = f.read()
            # comm 里可能有空格，从最后一个 ')' 之后开始切字段
            fields = st[st.rindex(")") + 2:].split()
            cpu = (int(fields[11]) + int(fields[12])) / HZ
            d = 0
            for tid in os.listdir(f"/proc/{pid}/task"):
                try:
                    with open(f"/proc/{pid}/task/{tid}/stat") as f:
                        tst = f.read()
                    if tst[tst.rindex(")") + 2] == "D":
                        d += 1
                except OSError:
                    pass
            out[int(pid)] = (cpu, d)
        except (OSError, ValueError, IndexError):
            continue
    return out


def vm():
    k = {}
    try:
        with open("/proc/vmstat") as f:
            for line in f:
                p = line.split()
                if p[0] in ("pgscan_direct", "allocstall_movable", "pgscan_kswapd"):
                    k[p[0]] = int(p[1])
    except OSError:
        pass
    return k


def psi():
    """`/proc/pressure/memory` 两行（some/full）的 `total=` 之和（微秒）。"""
    tot = 0
    try:
        with open("/proc/pressure/memory") as f:
            for line in f:
                for field in line.split():
                    if field.startswith("total="):
                        tot += int(field[len("total="):])
    except OSError:
        pass
    return tot


prev = (time.clock_gettime(M), time.clock_gettime(R), snapshot(), vm(), psi())
print("start drift=%+.6f（mono-raw，正常应缓慢漂移、无跳变）" % (prev[0] - prev[1]), flush=True)
while True:
    time.sleep(TICK)
    m, r = time.clock_gettime(M), time.clock_gettime(R)
    cur, v, p = snapshot(), vm(), psi()
    dm, dr = m - prev[0], r - prev[1]
    jump = abs(dm - dr) > 0.05 or dm > WFULL or dr > WFULL
    vmoved = any(v.get(k, 0) != prev[3].get(k, 0) for k in v)
    psimoved = p != prev[4]
    dcpu = sum(cpu - prev[2].get(pid, (cpu, 0))[0] for pid, (cpu, _) in cur.items())
    gap = dm - dcpu
    # ⚠️ 只在"这一步明显比 tick 长"或"时钟/回收变了"时打。
    #    不能用 gap > 小阈值：解码时主机线程大多阻塞在 stream 同步上，Δcpu 本来就≈0，
    #    那样会把每个 tick 都当成异常（踩过）。
    if jump or vmoved or psimoved or dm > WFULL:
        dlist = " ".join(f"pid{pid} cpu+{cpu - prev[2].get(pid, (cpu, 0))[0]:.2f} D{d}"
                         for pid, (cpu, d) in cur.items())
        print(
            f"{time.strftime('%H:%M:%S')} wall+{dm:.2f} cpu+{dcpu:.2f} gap+{gap:.2f} "
            f"| mono+{dm:.3f} raw+{dr:.3f} drift={m - r:+.6f} "
            f"| vmoved={vmoved} psimoved={psimoved} | {dlist}",
            flush=True,
        )
    prev = (m, r, cur, v, p)
