import threading

lock = threading.Lock()

def worker(num):
    with lock:
        print(f"Thread {num} acquired the lock")

threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
for t in threads:
    t.start()
for t in threads:
    t.join()