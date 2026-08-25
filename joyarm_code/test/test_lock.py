import sys, threading, time

print("=" * 50)
print("实际运行本脚本的解释器:", sys.version.split()[0], "@", sys.executable)
print("=" * 50)

def run_threads(target, n=8):
    ts = [threading.Thread(target=target) for _ in range(n)]
    for t in ts: t.start()
    for t in ts: t.join()

# 实验1：把竞态窗口撑开（读和写之间强制让出 GIL）
count = 0
def w1():
    global count
    for _ in range(1000):
        tmp = count
        time.sleep(0)        # 主动让出 GIL，窗口必然敞开
        count = tmp + 1
run_threads(w1)
print(f"实验1 撑开窗口:    {count} / 期望 8000")

# 实验2：函数调用夹在"读"和"写"之间
count = 0
def add_one(v):
    return v + 1
def w2():
    global count
    for _ in range(100000):
        count = add_one(count)
sys.setswitchinterval(1e-6)
run_threads(w2)
sys.setswitchinterval(0.005)
print(f"实验2 调用在窗口内: {count} / 期望 800000")

# 实验3：你的一行式原版，工作量加大到 8 x 1000000
count = 0
def w3():
    global count
    for _ in range(1000000):
        count = count + 1
run_threads(w3)
print(f"实验3 一行式巨量:   {count} / 期望 8000000")