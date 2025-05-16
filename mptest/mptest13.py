from multiprocessing import Process, JoinableQueue

def worker(q):
    while True:
        item = q.get()
        print(f'Processing {item}')
        q.task_done()  # 标记任务完成

if __name__ == '__main__':
    q = JoinableQueue()
    for i in range(2):  # 启动2个worker
        Process(target=worker, args=(q,), daemon=True).start()

    for item in ['A', 'B', 'C']:
        q.put(item)

    q.join()  # 阻塞直到所有任务完成
    print("All tasks completed")

# 学习一下并没有什么卵用的JoinableQueue。