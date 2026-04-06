import multiprocessing as mp
import time
import threading
import pytest
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

    @property
    def _internal_status(self):
        # A single underscore private member
        return "hidden_ok"

    def grab(self):
        # Local blocking simulated access
        time.sleep(0.1) 
        self._capture_count += 1
        return f"Frame_{self._capture_count}"

    def quick_grab(self):
        # Unblocked quick access for performance testing
        self._capture_count += 1
        return f"QFrame_{self._capture_count}"

    def trigger_crash(self):
        raise ValueError("Hardware SDK Crash Simulation!")


# --- Worker Function Definitions ---

def worker_standard(proxy: MpObjProxy):
    """Standard background worker serving RPC smoothly."""
    cam, lock = proxy()
    proxy.start_service_thread()
    time.sleep(60)

def worker_out_of_order(proxy: MpObjProxy, result_queue: mp.Queue):
    """Tests double invocation of __call__."""
    proxy()
    try:
        proxy()  # This should intentionally crash
        result_queue.put(False)
    except RuntimeError as e:
        result_queue.put(str(e))

def worker_transparent_assignment(proxy: MpObjProxy, result_queue: mp.Queue):
    """Tests that modifying internal engine variables forwards transparently."""
    cam, lock = proxy()
    
    # Intentionally write to a public variable that was previously a proxy core component
    proxy.target_obj = "mock_hacked"
    
    # The proxy.target_obj assignment should have succeeded seamlessly and updated the inner object's dict!
    if getattr(cam, "target_obj", None) == "mock_hacked":
        result_queue.put(True)
    else:
        result_queue.put(False)

def worker_lock_contention(proxy: MpObjProxy, barrier: mp.Barrier):
    cam, lock = proxy()
    proxy.start_service_thread()
    
    # Simulate a local worker grabbing sequentially holding the lock
    def capture_loop():
        barrier.wait() # Sync with Main before we start hogging the lock
        for _ in range(4): # Will hold lock for approx 0.4s
            proxy.grab()
            
    t = threading.Thread(target=capture_loop)
    t.start()
    t.join()


# --- Pytest Fixtures ---

@pytest.fixture
def managed_proxy():
    """Provides a ready-to-use Proxy instance connected to a background Worker Process."""
    proxy = MpObjProxy(TestCamera, fps=60)
    worker = mp.Process(target=worker_standard, args=(proxy,))
    worker.start()
    
    proxy.wait_handshake()
    yield proxy
    
    # Teardown
    worker.terminate()
    worker.join()


# --- Unit Tests: Integrity & Defensive Shielding ---

def test_out_of_order_invocation():
    """Validates the proxy lifecycle defenses against misuse and out-of-order execution."""
    proxy = MpObjProxy(TestCamera)
    
    # 1. Unpickled Host Call
    with pytest.raises(RuntimeError, match="cannot be '__call__'ed before being pickled"):
        proxy()
        
    # 2. Start Service Thread in Host
    with pytest.raises(RuntimeError, match="Cannot start RPC service in Host"):
        proxy.start_service_thread()
        
    # 3. Premature Interaction 
    with pytest.raises(RuntimeError, match="Proxy is not initialized. Call wait_handshake"):
        res = proxy.grab()
        
    # 4. Double Initialization (Worker side logic)
    q = mp.Queue()
    worker = mp.Process(target=worker_out_of_order, args=(proxy, q))
    worker.start()
    
    msg = q.get(timeout=5)
    assert "has already been called before" in msg, "Proxy did not adequately defend double instantiation."
    
    worker.terminate()
    worker.join()

def test_proxy_namespace_shielding(managed_proxy):
    """Verifies that internal attributes are kept entirely local and unresolvable proxies error out gracefully."""
    # 1. Invalid Internal Query -> Blocks immediately, no RPC dispatch!
    with pytest.raises(AttributeError, match="proxy has no internal attribute '_pxy_typo_attr_'"):
        _ = managed_proxy._pxy_typo_attr_
        
    # 2. Local State modification -> Absorbed into local wrapper dictionary silently
    managed_proxy._pxy_temp_ = "test_shield"
    assert managed_proxy._pxy_temp_ == "test_shield"

def test_transparent_assignment(managed_proxy):
    """Validates public namespace is completely free and gracefully redirects everything."""
    # Main Process attempt: Host refuses to assign things not explicitly listed remotely
    with pytest.raises(AttributeError, match="has no assignable attribute 'target_obj'"):
        managed_proxy.target_obj = "hacked_from_main"
        
    # Worker Process attempt: Worker should freely and naturally set properties on the dummy object
    q = mp.Queue()
    proxy = MpObjProxy(TestCamera)
    worker = mp.Process(target=worker_transparent_assignment, args=(proxy, q))
    worker.start()
    
    msg = q.get(timeout=5)
    assert msg is True, "Proxy intercepted public target_obj property assignment erroneously!"
    
    worker.terminate()
    worker.join()

def test_remote_private_attributes(managed_proxy):
    """Verifies that valid single '_' private attributes of the target are correctly forwarded now."""
    assert managed_proxy._internal_status == "hidden_ok"


# --- Unit Tests: RPC & Behaviors ---

def test_propery_access(managed_proxy):
    assert managed_proxy.fps == 60
    managed_proxy.fps = 120
    assert managed_proxy.fps == 120

def test_method_call(managed_proxy):
    res = managed_proxy.grab()
    assert res == "Frame_1"
    
def test_exception_persistence(managed_proxy):
    with pytest.raises(ValueError, match="Hardware SDK Crash Simulation!"):
        managed_proxy.trigger_crash()

def test_functor_property_behavior(managed_proxy):
    # Verify we get an isolated copy of functor objects
    functor_copy = managed_proxy.functor_prop
    assert functor_copy.secret_val == 42
    
def test_mp_eval_remote_execution(managed_proxy):
    # Execute actual nested execution on the remote side bypassing the copy limitation
    res = managed_proxy.mp_eval("self.functor_prop(100)")
    assert res == 142
    
def test_mp_eval_nested_setattr(managed_proxy):
    # Modify deep properties via kwargs
    managed_proxy.mp_eval("setattr(self.functor_prop, 'secret_val', new_val)", new_val=999)
    assert managed_proxy.mp_eval("self.functor_prop.secret_val") == 999

def test_lock_contention_and_queuing():
    """Validates that concurrent remote and local Proxy access correctly blocks behind rpc_lock."""
    proxy = MpObjProxy(TestCamera)
    barrier = mp.Barrier(2)
    
    worker = mp.Process(target=worker_lock_contention, args=(proxy, barrier))
    worker.start()
    proxy.wait_handshake()
    
    barrier.wait() # Synchro: Worker is starting the local 0.4s block!
    time.sleep(0.05) 
    
    t0 = time.time()
    res = proxy.grab()
    t1 = time.time()
    
    assert res.startswith("Frame_")
    # Due to contention, taking the lock could be anywhere between 0.1 and 0.4s 
    # depending on timing of RPC dispatch, but is definitely not 0!
    assert (t1 - t0) >= 0.05, "RPC did not seem to be blocked by Worker's local proxy execution!"
    
    worker.terminate()
    worker.join()

# --- Performance Benchmarks ---

def test_performance_rpc_getattr(managed_proxy):
    """Benchmarks overhead of a raw property access via IPC."""
    iterations = 1000
    t0 = time.perf_counter()
    
    for _ in range(iterations):
        _ = managed_proxy.fps
        
    t1 = time.perf_counter()
    total_time = t1 - t0
    ops_per_sec = iterations / total_time
    
    print(f"\n[PERFORMANCE] RPC getattr: {ops_per_sec:.2f} Ops/sec | Time per call: {(total_time/iterations)*1000:.3f} ms")
    
    overhead_ms = (total_time/iterations) * 1000
    assert overhead_ms < 5.0, f"Performance Degraded! Taking {overhead_ms}ms per attribute access."

def test_performance_rpc_fast_call(managed_proxy):
    """Benchmarks overhead of executing a method via IPC."""
    iterations = 1000
    t0 = time.perf_counter()
    
    for _ in range(iterations):
        _ = managed_proxy.quick_grab()
        
    t1 = time.perf_counter()
    total_time = t1 - t0
    ops_per_sec = iterations / total_time
    
    print(f"\n[PERFORMANCE] RPC method call: {ops_per_sec:.2f} Ops/sec | Time per call: {(total_time/iterations)*1000:.3f} ms")
    
    overhead_ms = (total_time/iterations) * 1000
    assert overhead_ms < 5.0, f"Performance Degraded! Taking {overhead_ms}ms per cross-line method evaluation."

def worker_performance_local(proxy: MpObjProxy, result_queue: mp.Queue):
    """Worker designed purely to benchmark internal native access latency."""
    cam, lock = proxy()
    iterations = 50000
    
    # 1. Benchmark Local Proxy Access (which wraps with rpc_lock)
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = proxy.fps
    t1 = time.perf_counter()
    locked_time = t1 - t0
    
    # 2. Benchmark Local Native Access (Lock-Free raw access)
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = cam.fps
    t1 = time.perf_counter()
    bare_time = t1 - t0
    
    result_queue.put((iterations, locked_time, bare_time))

def test_performance_local_overhead():
    """Benchmarks the latency of local proxy interception vs un-intercepted bare access in the Worker."""
    proxy = MpObjProxy(TestCamera)
    q = mp.Queue()
    worker = mp.Process(target=worker_performance_local, args=(proxy, q))
    worker.start()
    
    iterations, locked_time, bare_time = q.get(timeout=5)
    
    print(f"\n[PERFORMANCE] Local Proxy (Locked): {iterations/locked_time:,.2f} Ops/sec | Time per call: {(locked_time/iterations)*1e6:.3f} us")
    print(f"[PERFORMANCE] Local Bare  (Lock-Free): {iterations/bare_time:,.2f} Ops/sec | Time per call: {(bare_time/iterations)*1e6:.3f} us")
    
    worker.terminate()
    worker.join()
