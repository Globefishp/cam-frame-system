import requests
from multiprocessing import Process, set_start_method
import pickle
import os

def child_process_use_session(session):
    try:
        pid = os.getpid()
        print(f"Child process {pid} attempting to use session...")
        response = session.get('https://httpbin.org/get')
        print(f"Child process {pid} request successful!")
        print(f"Session headers in child: {session.headers}")
        print(response.json())
    except Exception as e:
        print(f"Child process failed to use session: {str(e)}")

def child_process_with_pickled_session(pickled_session):
    try:
        pid = os.getpid()
        print(f"Child process {pid} attempting to unpickle session...")
        session = pickle.loads(pickled_session)
        print(f"Unpickle successful in {pid}, attempting request...")
        response = session.get('https://httpbin.org/get')
        print(f"Child process {pid} request with unpickled session successful!")
        print(f"Unpickled session headers: {session.headers}")
        print(response.json())
    except Exception as e:
        print(f"Child process failed to use pickled session: {str(e)}")

def child_create_session(headers):
    try:
        pid = os.getpid()
        session = requests.Session()
        session.headers.update(headers)
        print(f"Child process {pid} created new session")
        response = session.get('https://httpbin.org/get')
        print(f"Child-created session request successful in {pid}!")
        print(f"New session headers: {session.headers}")
        print(response.json())
    except Exception as e:
        print(f"Child failed to create session: {str(e)}")

if __name__ == '__main__':
    # 明确设置启动方法以便在不同平台上测试
    set_start_method('spawn')  # 可以改为'fork'在Unix上测试
    
    print(f"Main process PID: {os.getpid()}")
    
    # 在主进程创建Session
    main_session = requests.Session()
    main_session.headers.update({'X-Test': 'main-process'})
    print(f"Main session headers: {main_session.headers}")
    
    # 测试直接传递Session对象
    print("\n1. Testing direct session passing:")
    p1 = Process(target=child_process_use_session, args=(main_session,))
    p1.start()
    p1.join()
    
    # 测试pickle序列化后传递
    print("\n2. Testing pickled session passing:")
    try:
        pickled = pickle.dumps(main_session)
        p2 = Process(target=child_process_with_pickled_session, args=(pickled,))
        p2.start()
        p2.join()
    except Exception as e:
        print(f"Main process failed to pickle session: {str(e)}")
    
    # 替代方案：在新进程中创建新Session
    print("\n3. Testing creating new session in child process:")
    p3 = Process(target=child_create_session, args=({'X-Test': 'child-process'},))
    p3.start()
    p3.join()
    
    # 测试修改主进程session是否影响子进程
    print("\n4. Testing session modification:")
    main_session.headers.update({'X-Modified': 'true'})
    p4 = Process(target=child_process_use_session, args=(main_session,))
    p4.start()
    p4.join()