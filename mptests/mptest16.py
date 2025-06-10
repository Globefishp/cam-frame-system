import threading
import time
import statistics

def test_notify_impact():
    """测试 notify 对 Condition.wait_for() 延迟的影响"""
    cond = threading.Condition()
    results = {'with_notify': [], 'without_notify': []}
    test_count = 1000  # 测试次数

    def worker(use_notify, results_list):
        flag = [False]
        
        def setter():
            with cond:
                flag[0] = True
                if use_notify:
                    cond.notify()  # 只有这个分支会发送通知

        # 启动测试线程
        t = threading.Thread(target=setter)
        t.start()

        # 测量等待时间
        with cond:
            start_time = time.perf_counter_ns()
            cond.wait_for(lambda: flag[0])
            end_time = time.perf_counter_ns()
        
        results_list.append(end_time - start_time)  
        t.join()

    # 运行测试
    for _ in range(test_count):
        # 测试有notify的情况
        worker(use_notify=True, results_list=results['with_notify'])
        
        # 测试无notify的情况
        worker(use_notify=False, results_list=results['without_notify'])

    # 打印统计结果
    print(f"测试次数: {test_count}")
    print("有 notify 的情况:")
    print(f"  平均延迟: {statistics.mean(results['with_notify']):.3f} ns")
    print(f"  最大延迟: {max(results['with_notify']):.3f} ns")
    print(f"  最小延迟: {min(results['with_notify']):.3f} ns")
    
    print("\n无 notify 的情况:")
    print(f"  平均延迟: {statistics.mean(results['without_notify']):.3f} ns")
    print(f"  最大延迟: {max(results['without_notify']):.3f} ns")
    print(f"  最小延迟: {min(results['without_notify']):.3f} ns")

if __name__ == '__main__':
    test_notify_impact()