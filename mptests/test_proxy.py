import multiprocessing as mp
import time
import threading
from utils.mp_obj_proxy import MpObjProxy

# --- Dummy Objects to Test Functor/Nested Behavior ---
class InnerFunctor:
    def __init__(self):
        self.secret_val = 42
        
    def __call__(self, offset):
        return self.secret_val + offset

class TestCamera:
    def __init__(self, fps=30):
        self._fps = fps
        self._capture_count = 0
        self.functor_prop = InnerFunctor()

    @property
    def fps(self):
        return self._fps

    @fps.setter
    def fps(self, v):
        self._fps = v

    def grab(self):
        # Simulate hardware capture delay
        time.sleep(0.5)
        self._capture_count += 1
        return f"Frame_{self._capture_count}"

    def trigger_crash(self):
        raise ValueError("Hardware SDK Crash Simulation!")


# --- Worker Process Logic ---
def worker_process(proxy: MpObjProxy):
    # RISK 1: Double Call Defense
    proxy()
    print("[Worker] Proxy unpacked successfully.")
    
    try:
        proxy()
        print("[Worker] FATAL: Should not be able to call twice!")
    except RuntimeError:
        print("[Worker] Defense Validation: Cannot call proxy twice (SUCCESS).")

    # Start the daemon RPC service thread
    print("[Worker] Starting RPC Service Daemon Thread...")
    proxy.mp_start_service_thread()
    
    # Simulate internal threaded Capture Loop that shares the target_obj
    def capture_loop():
        for i in range(2):
            # RISK 2: Thread-Safe Local Forwarding vs Lock-Free
            # Since grab() runs via the proxy here, it holds the proxy.rpc_lock for 0.5s!
            print(f"\n   [Worker-Thread] Grabbing frame using thread-safe proxy...")
            f = proxy.grab() 
            print(f"   [Worker-Thread] Got {f} locally!")
            
            # Lock-Free access (Bypassing proxy serialization layer):
            _ = proxy.target_obj.functor_prop.secret_val
            
    capture_thread = threading.Thread(target=capture_loop)
    capture_thread.start()
    
    # Block worker alive
    capture_thread.join()
    # Wait to serve final commands before shutting down
    time.sleep(2)


# --- Main Process Logic ---
if __name__ == "__main__":
    print("\n======= STARTING MP PROXY TEST =======")
    print("[Main] Creating Proxy Host...")
    proxy = MpObjProxy(TestCamera, fps=60)

    # Note: Double Pickle Defense isn't easily tested here because passing the host 
    # to Process Pool will permanently kill its handles, but the code architecture protects it.

    worker = mp.Process(target=worker_process, args=(proxy,))
    worker.start()

    print("[Main] Waiting for handshake...")
    proxy.wait_handshake()
    print(f"[Main] Handshake complete! Remote methods detected: {proxy._remote_methods}")

    # RISK 3: Test Basic Prop & Method & Exception Serialization
    print("\n[Main] Testing Initial Property Value:")
    print(f"       proxy.fps = {proxy.fps}")

    print("\n[Main] Testing Method Call (This will QUEUE against Worker's local capture loop lock!):")
    # This should wait until the Worker finishes its current grab!
    start_time = time.time()
    f = proxy.grab()
    print(f"       proxy.grab() = {f}. Elapsed wait time: {time.time() - start_time:.2f}s")

    print("\n[Main] Testing Property Setter:")
    proxy.fps = 120
    print(f"       proxy.fps (after set) = {proxy.fps}")

    print("\n[Main] Testing Exception Persistence:")
    try:
        proxy.trigger_crash()
    except ValueError as e:
        print(f"       Caught expected error reflecting true exception type: [{type(e).__name__}] {e}")

    # RISK 4: Eval for Nested Objects and Functors (Callable misclassification defense)
    print("\n[Main] Testing Nested properties & Functor Handling:")
    
    # Direct access: proxy.functor_prop is a property! It will return an isolated COPY of InnerFunctor!
    functor_copy = proxy.functor_prop
    print(f"       Isolated Object Copy .secret_val: {functor_copy.secret_val}")
    
    # RISK 5: Remote call via mp_eval to directly call the functor ON the remote end!
    eval_res = proxy.mp_eval("self.functor_prop(100)")
    print(f"       Remote mp_eval result (42 + 100): {eval_res}")

    print("\n[Main] Testing mp_eval with kwargs for nested remote attribute execution:")
    proxy.mp_eval("setattr(self.functor_prop, 'secret_val', new_secret)", new_secret=999)
    new_secret = proxy.mp_eval("self.functor_prop.secret_val")
    print(f"       Remote nested value successfully modified natively: {new_secret}")


    print("\n[Main] Testing finished. Shutting down worker...")
    worker.terminate()
    worker.join()
    print("======= TEST END =======")
