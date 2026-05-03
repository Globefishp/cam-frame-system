# utils/mp_obj_proxy.py
# Author: Google Gemini 3.1 pro, modified & cleanup by Haiyun Huang (260406)

from inspect import getattr_static
import threading
import multiprocessing as mp
from multiprocessing.connection import Connection
import inspect
from typing import Any, Tuple, Dict, Type, Optional, Set, Callable

HAS_GETMEMBERS_STATIC = hasattr(inspect, 'getmembers_static')

class MpObjProxy:
    """
    MpObjProxy is a factory class, whose instance allows instantiating an object
    in subprocess and establishing pipe connection with instance in host process.
    Once connection established, the proxy instance in host process behaves like the 
    object born in the subprocess, via RPC through internal pipes.

    Limitations: 
        - Multi-processing must in spawn mode: mp.set_start_method('spawn', force=True)
        - Only allow one to one RPC; 
        - Properties/methods dynamically created after `__init__` will not be 
          proxied automatically. Use `mp_rescan` to update member cache (see below)
          Properties not in `dict` (i.e. `__getattr__` or `__getattribute__`) cannot
          be detected.
        - returned property object is a copy, not sync'ed with the remote object.
          So any access to `self.obj.obj_value` should be forwarded by 
          `@property self.fwd_obj_value` or using `mp_eval` (see below)
    
    Usage:
        # 1. In main process
        proxy = MpObjProxy(TargetClass, *arg, **kwargs)
        - This will commit to TargetClass(*arg, **kwargs) in subprocess.

        # 2. Pass to subprocess, implicitly serialized
        sp = MpProcessClass(target_factory=proxy)
        sp.start()

        # 3. Instantiate target object in subprocess
        target_obj, rpc_lock = proxy()
        proxy.start_service_thread()

        # 4. Wait for initialization (# 3) in Main Process
        proxy.wait_handshake()

        # 5. In subprocess, use proxy as the target_obj with implicit Lock(), 
             or use target_obj without Lock() (with care, could conflict with RPC)

        # 6. In main process, use proxy as the target_obj
        Limitations: 
            - _pxy_<name>_ properties and functions like `wait_handshake`, 
              `start_service_thread`, `mp_rescan`, `mp_eval`, `__call__` are 
              reserved names by proxy;
            - returned properties are copies, not sync'ed with the remote object.
              using `mp_eval` to do complex things.
        
        # 7. If any methods change the member list of target_obj, call mp_rescan()
             to let new/deleted members be proxied/unproxied.
    """

    def __init__(self, target_cls: Type, *args, **kwargs):
        # Internal variables are all in _pxy_<name>_ format
        self._pxy_cls_ = target_cls
        self._pxy_init_args_ = args
        self._pxy_init_kwargs_ = kwargs

        # --- Handshake Pipe (Subprocess -> Main) ---
        self._pxy_init_rx_, self._pxy_init_tx_ = mp.Pipe(duplex=False)

        # --- Dual Unidirectional Pipes for RPC ---
        self._pxy_owner_rx_, self._pxy_main_tx_ = mp.Pipe(duplex=False)
        self._pxy_main_rx_, self._pxy_owner_tx_ = mp.Pipe(duplex=False)
        # owner_: owner for the object.

        # --- Introspection Cache ---
        self._pxy_remote_methods_: Set[str] = set() 
        self._pxy_remote_properties_: Set[str] = set()
        
        # Determine if handshake has completed
        self._pxy_is_ready_ = False
        # Flag for pickled state
        self._pxy_pickled_ = False

        # --- Server State (Available after __call__) ---
        self.__dict__['_pxy_target_obj_'] = None
        self.__dict__['_pxy_rpc_lock_'] = None
        self._pxy_service_thread_ = None
        self._pxy_shutdown_event_ = None

    def __getstate__(self) -> dict:
        """
        Invoked when the proxy is pickled, typically in subprocess.start()
        """
        state = self.__dict__.copy()
        
        # Remove pipe ends unwanted for subprocess to avoid file descriptor leakage
        state.pop('_pxy_init_rx_', None)
        state.pop('_pxy_main_rx_', None)
        state.pop('_pxy_main_tx_', None)
        # Mark as pickled
        state['_pxy_pickled_'] = True
        
        return state

    # __setstate__ is not needed since default behavior is to update __dict__

    # === Main process ===
    def wait_handshake(self) -> None:
        """
        Invoked in Main Process.
        Waits for the child process to call this proxy and report the class signature.
        """
        # Close the transmission ends that only belong to the subprocess immediately.
        # 'let it crash': they should exist.
        self._pxy_init_tx_.close()
        del self._pxy_init_tx_
        self._pxy_owner_tx_.close()
        del self._pxy_owner_tx_
        self._pxy_owner_rx_.close()
        del self._pxy_owner_rx_

        try:
            msg = self._pxy_init_rx_.recv()
        except EOFError as e:
            raise RuntimeError("Target process exited unexpectedly before finishing handshake.") from e

        self._pxy_init_rx_.close() # Handshake complete, cleanup
        del self._pxy_init_rx_

        self._pxy_remote_methods_ = msg["methods"]
        self._pxy_remote_properties_ = msg["properties"]
        self._pxy_is_ready_ = True

    def _ipc_send_and_wait(self, action: str, name: str, args: tuple = (), kwargs: dict = None) -> Any:
        """Internal helper to dispatch RPC command and await serialization target."""
        if not self._pxy_is_ready_:
            raise RuntimeError(f"MpObjProxy is not ready! Call wait_handshake() first. Attempted access: {name}")

        request = (action, name, args, kwargs or {})
        self._pxy_main_tx_.send(request)
        
        success, payload = self._pxy_main_rx_.recv()
        
        if not success:
            raise payload
        return payload

    def mp_eval(self, stmt: str, **kwargs) -> Any:
        """
        Evaluates a string expression dynamically in the subprocess.
        'self' will be mapped to the remote target_obj in RPC.
        Example: proxy.mp_eval("self.sensor_sys.get_version(mode=1)")
                 proxy.mp_eval("setattr(self, 'sensor_value', value)", value=1)

        :return: Any serializable object from RPC pipe.
        """
        return self._ipc_send_and_wait("eval", "mp_eval", args=(stmt,), kwargs=kwargs)
    
    def mp_rescan(self) -> None:
        """
        Rescan the target object, update properties and methods list.
        Useful when dynamically added/removed attributes in target object.
        """
        payload = self._ipc_send_and_wait("rescan", "mp_rescan")
        self._pxy_remote_methods_ = payload["methods"]
        self._pxy_remote_properties_ = payload["properties"]

    # === Subprocess ===
    def __call__(self) -> Tuple[Any, threading.Lock]:
        """
        Should be invoked only once inside one subprocess.
        Instantiates the actual object, inspects it, signals the main process.
        
        :returns `(target_obj, rpc_lock)`: The bare target instances for maximum performance.
            rpc_lock is acquired when background thread handled with RPC command.
        """
        if not self._pxy_pickled_:
            raise RuntimeError("MpObjProxy cannot be '__call__'ed before being pickled.")
        if self._pxy_is_ready_:
            # weak check, just defend some misuse cases.
            raise RuntimeError("MpObjProxy has already been called before.")

        # Instantiate target object without __setattr__ interference
        self.__dict__['_pxy_target_obj_'] = self._pxy_cls_(*self._pxy_init_args_, **self._pxy_init_kwargs_)
        # Initialize RPC resources
        self.__dict__['_pxy_rpc_lock_'] = threading.Lock()
        self._pxy_shutdown_event_ = threading.Event()

        # Introspection and send categorized member names back to Main Process
        payload = self._inspect_target_obj()

        self._pxy_init_tx_.send(payload)
        self._pxy_init_tx_.close() # Close init_tx as it's no longer needed
        del self._pxy_init_tx_

        self._pxy_is_ready_ = True
        return self._pxy_target_obj_, self._pxy_rpc_lock_

    if HAS_GETMEMBERS_STATIC:
        def _inspect_target_obj(self) -> Dict[str, Set[str]]:
            """
            Internal helper for introspecting the target object by categorizing into
            methods (evoke `__call__`) and properties (evoke `getattr`/`setattr`).
            Python 3.11+
            """
            methods = set()
            properties = set()

            for name, static_member in inspect.getmembers_static(self._pxy_target_obj_):
                # getmembers_static won't exec user code that may raise.
                if name.startswith('__'):  # Ignore magic methods.
                    continue
                    
                # inspect.isroutine safely captures methods, functions, staticmethods, and C-builtins, 
                # avoiding misclassifying object with __call__.
                if inspect.isroutine(static_member):
                    methods.add(name)
                else:
                    properties.add(name)

            return {"methods": methods, "properties": properties}
    else:
        def _inspect_target_obj(self) -> Dict[str, Set[str]]:
            """
            Internal helper for introspecting the target object by categorizing into
            methods (evoke `__call__`) and properties (evoke `getattr`/`setattr`).
            Python 3.2+
            """
            methods = set()
            properties = set()

            for name in dir(self._pxy_target_obj_): 
                # `inspect.getmembers` will use `getattr`, which will trigger getter 
                # function execution for a property. If it raise an error, inspection
                # will fail. `dir` is a workaround.
                if name.startswith('__'):  # Ignore magic methods.
                    continue
                
                # `getattr_static` will not invoke descriptor.__get__, thus won't raise.
                static_member = getattr_static(self._pxy_target_obj_, name)
                    
                # inspect.isroutine safely captures methods, functions, staticmethods, and C-builtins, 
                # avoiding misclassifying object with __call__.
                if inspect.isroutine(static_member):
                    methods.add(name)
                else:
                    properties.add(name)

            return {"methods": methods, "properties": properties}

    def _rpc_service_step(self, block: bool = True) -> bool:
        """
        Single atomic step of RPC execution in subprocess.
        Executes incoming commands holding `self._pxy_rpc_lock_`.
        Can be packed into a subthread.

        :param block: Whether to block until a command is received.
        :return: True if a command was executed, False otherwise.
        """
        if not block and not self._pxy_owner_rx_.poll():
            return False
            
        try:
            req = self._pxy_owner_rx_.recv()
        except EOFError:
            return False

        action, name, args, kwargs = req

        with self._pxy_rpc_lock_:
            try:
                if action == "getattr":
                    res = getattr(self._pxy_target_obj_, name)
                    self._pxy_owner_tx_.send((True, res))
                elif action == "setattr":
                    setattr(self._pxy_target_obj_, name, args[0])
                    self._pxy_owner_tx_.send((True, None))
                elif action == "call":
                    func = getattr(self._pxy_target_obj_, name)
                    res = func(*args, **kwargs)
                    self._pxy_owner_tx_.send((True, res))
                elif action == "eval":
                    env = {"self": self._pxy_target_obj_}
                    env.update(kwargs)
                    res = eval(args[0], {}, env) # eval(expr, globals, locals)
                    self._pxy_owner_tx_.send((True, res))
                elif action == "rescan":
                    payload = self._inspect_target_obj()
                    self._pxy_owner_tx_.send((True, payload))
                else:
                    self._pxy_owner_tx_.send((False, RuntimeError(f"Unknown Action: {action}")))
            except Exception as e:
                self._pxy_owner_tx_.send((False, e))
                
        return True

    def start_service_thread(self) -> None:
        """Start a daemon thread to serve RPC commands continuously."""
        if not self._pxy_pickled_:
             raise RuntimeError("Cannot start RPC service in Host. It must be called in subprocess after being pickled.")
        if not self._pxy_is_ready_:
             raise RuntimeError("Subprocess proxy is not initialized. Invoke __call__() first.")
             
        def _loop():
            while not self._pxy_shutdown_event_.is_set():
                if self._pxy_owner_rx_.poll(0.01): # 10ms timeout check
                    self._rpc_service_step(block=True)
                    
        self._pxy_service_thread_ = threading.Thread(target=_loop, daemon=True, name=f"RPCProxy_{self._pxy_cls_.__name__}")
        self._pxy_service_thread_.start()

    # === Executor ===
    def __getattr__(self, name: str) -> Any:
        """
        Forward dynamic attribute access to internal proxy properties, local object
        properties or IPC.
        
        Proxy properties are "_pxy_" + property_name + "_".
        Other name will be forwarded to local object or IPC automatically.
        """
        # Handle proxy's own internal properties
        if (name.startswith('_pxy_') and name.endswith('_')) or (
            name.startswith('__') and name.endswith('__')):
            # Python will search __dict__, if not exist, call __getattr__, thus raise directly.
            raise AttributeError(f"'{self.__class__.__name__}' proxy has no internal attribute '{name}'")

        # Local access by owner
        # Inside get/setattr, use __dict__.get to avoid some case that __dict__ 
        # is not initialized as we thought (e.g. in `pickle`)
        if self.__dict__.get('_pxy_pickled_', False):
            if not self.__dict__.get('_pxy_is_ready_', False):
                raise RuntimeError("Local proxy is not initialized. Call __call__() first.")

            attr = getattr(self._pxy_target_obj_, name)
            if inspect.isroutine(attr):
                # By default, call method through proxy is thread-safe (wrapped by rpc_lock).
                def _locked_call(*args, **kwargs):
                    with self._pxy_rpc_lock_:
                        return attr(*args, **kwargs)
                return _locked_call
            else:
                with self._pxy_rpc_lock_:
                    return attr

        # Access through RPC
        if not self.__dict__.get('_pxy_pickled_', False):
            if not self.__dict__.get('_pxy_is_ready_', False):
                raise RuntimeError("Proxy is not initialized. Call wait_handshake() first.")

            if name in self.__dict__.get('_pxy_remote_properties_', set()):
                return self._ipc_send_and_wait("getattr", name)
                
            if name in self.__dict__.get('_pxy_remote_methods_', set()):
                return _RemoteMethodCallable(name, self)

            #TODO: if not found, could be dynamically created in remote object,
            # re-inspect and try again? or directly fwd to remote and wait for error?

        raise AttributeError(f"Remote object '{self._pxy_cls_.__name__}' has no public attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Forward dynamic attribute access to internal proxy properties, local object
        properties or IPC.
        
        Proxy properties are "_pxy_" + property_name + "_".
        Other name will be forwarded to local object or IPC automatically.
        
        Note: Internal access to special variables not in `_pxy_<>_` format
        should modify __dict__ directly.
        """
        # Handle proxy's own internal properties
        if name.startswith('_pxy_') and name.endswith('_'):
            self.__dict__[name] = value
            return
            
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError("Cannot remotely assign to magic methods.")
            
        # Local access (Thread-safe by default)
        if self.__dict__.get('_pxy_pickled_', False):
            if not self.__dict__.get('_pxy_is_ready_', False):
                raise RuntimeError("Local proxy is not initialized. Call __call__() first.")

            with self._pxy_rpc_lock_:
                setattr(self._pxy_target_obj_, name, value)
            return

        # Access through RPC
        if not self.__dict__.get('_pxy_pickled_', False):
            if not self.__dict__.get('_pxy_is_ready_', False):
                raise RuntimeError("Proxy is not initialized. Call wait_handshake() first.")

            # Write to remote property, but not _remote_methods (Prohibit monkey patch)
            if name in self.__dict__.get('_pxy_remote_properties_', set()):
                # Send setattr instruction
                self._ipc_send_and_wait("setattr", name, args=(value,))
                return

        raise AttributeError(f"Remote object '{self._pxy_cls_.__name__}' has no assignable attribute '{name}'")


class _RemoteMethodCallable:
    """
    A helper class for MpObjProxy to make remote methods callable.
    When invoked, it packages args and kwargs and triggers the IPC 'call'.
    """
    def __init__(self, name: str, proxy_instance: MpObjProxy):
        self._name = name
        self._proxy = proxy_instance

    def __call__(self, *args, **kwargs):
        return self._proxy._ipc_send_and_wait("call", self._name, args, kwargs)
