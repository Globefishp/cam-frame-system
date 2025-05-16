import multiprocessing
import time
import statistics
from multiprocessing import Process, Condition, Value

def worker(cond, flag, use_notify, result_queue):
    """子进程工作函数"""
    with cond:
        start_time = time.perf_counter_ns()
        cond.wait_for(lambda: flag.value == 1)
        end_time = time.perf_counter_ns()
    
    # 将结果放入队列(转换为毫秒)
    result_queue.put((end_time - start_time) / 1e3)

def test_multiprocess_notify_impact():
    """测试多进程环境下 notify 对 wait_for 的影响"""
    test_count = 100  # 测试次数(进程创建开销大，不宜太多)
    results = {'with_notify': [], 'without_notify': []}
    
    for _ in range(test_count):
        # 测试有notify的情况
        cond = Condition()
        flag = Value('i', 0)
        result_queue = multiprocessing.Queue()
        
        p = Process(target=worker, args=(cond, flag, True, result_queue))
        p.start()
        
        # 主进程修改条件
        time.sleep(0.01)  # 确保子进程先进入等待
        with cond:
            flag.value = 1
            cond.notify()
        
        p.join()
        results['with_notify'].append(result_queue.get())
        
        # 测试无notify的情况
        cond = Condition()
        flag = Value('i', 0)
        result_queue = multiprocessing.Queue()
        
        p = Process(target=worker, args=(cond, flag, False, result_queue))
        p.start()
        
        time.sleep(0.01)  # 确保子进程先进入等待
        with cond:
            flag.value = 1  # 不调用notify
        
        p.join()
        results['without_notify'].append(result_queue.get())

    # 打印统计结果
    print(f"测试次数: {test_count}")
    print("有 notify 的情况:")
    print(f"  平均延迟: {statistics.mean(results['with_notify']):.3f} us")
    print(f"  最大延迟: {max(results['with_notify']):.3f} us")
    print(f"  最小延迟: {min(results['with_notify']):.3f} us")
    
    print("\n无 notify 的情况:")
    print(f"  平均延迟: {statistics.mean(results['without_notify']):.3f} us")
    print(f"  最大延迟: {max(results['without_notify']):.3f} us")
    print(f"  最小延迟: {min(results['without_notify']):.3f} us")

if __name__ == '__main__':
    # Windows下需要这行代码
    multiprocessing.freeze_support()
    test_multiprocess_notify_impact()