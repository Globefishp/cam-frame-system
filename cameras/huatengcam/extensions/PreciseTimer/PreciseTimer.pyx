# file: PrecisionTimer.pyx
from libc.stdint cimport uint16_t
import ctypes

# Import the C declarations from our header file
cdef extern from "timer_core.h":
    # Define our C function pointer type again for Cython
    ctypedef void* HANDLE
    ctypedef int (*trigger_func_ptr_t)(HANDLE, uint16_t*)

    # Opaque struct pointer and functions
    ctypedef struct TimerState:
        pass
    
    TimerState* timer_core_create(
        long long interval_ns,
        trigger_func_ptr_t trigger_func,
        HANDLE hCamera,
        int busy_wait_ns,
        int priority,
        int debug_print
    )
    void timer_core_start(TimerState* state)
    void timer_core_stop(TimerState* state)
    void timer_core_destroy(TimerState* state)

# The Python-facing class
cdef class PreciseTimer:
    cdef TimerState* state  # Pointer to our C state object
    cdef public bint is_running

    def __cinit__(self, 
                  double interval_s, 
                  object c_trigger_func, 
                  object hCamera, # Python obj of HANDLE
                  int busy_wait_us=2000, 
                  int priority=0,
                  bint debug_print=False):
        """
        Initializes the high-precision timer.

        Parameters
        ----------
        interval_s : float
            The desired time interval between triggers, in seconds.
        c_trigger_func : ctypes function object
            The C-level trigger function to call. This should be a function
            pointer from a loaded library, e.g. function object from 
            `C.windll.LoadLibrary()`. The expected signature is 
            `int trigger_func(HANDLE hCamera, uint16_t*)`.
        hCamera : ctypes.c_void_p
            The integer handle to the camera, which will be passed to the
            trigger function.
        busy_wait_us : int, optional
            The duration in microseconds (µs) for the final busy-wait loop.
            This determines the trade-off between CPU usage and timing precision.
            It should be slightly longer than the system's timer resolution
            (which is set to 1ms by this timer). A value of 2000 (2ms) is a
            good starting point. Default is 2000.
        priority : int, optional
            The priority of the internal timer thread.
            - 0: Normal (default)
            - 1: Above Normal
            - 2: Highest
            - 3: Time Critical (use with caution, can affect system stability)
        debug_print : bool, optional
            Whether to print c-level debug messages to stdout.
            Default is False.
        """
        # 1. Extract the raw C function address from the ctypes object
        cdef trigger_func_ptr_t func_ptr
        func_addr = <size_t>ctypes.cast(c_trigger_func, ctypes.c_void_p).value
        func_ptr = <trigger_func_ptr_t>func_addr

        # 2. Call the C 'constructor' to create and initialize the state
        interval_ns = <long long>(interval_s * 1_000_000_000)
        busy_wait_ns = busy_wait_us * 1000

        cdef HANDLE hCamera_ptr
        if isinstance(hCamera, ctypes.c_void_p):
            hCamera_ptr = <HANDLE><size_t>hCamera.value
        else:
            raise TypeError("hCamera must be ctypes.c_void_p instance.")
        
        self.state = timer_core_create(interval_ns, func_ptr, hCamera_ptr, busy_wait_ns, priority, int(debug_print))
        if self.state is NULL:
            raise MemoryError("Failed to allocate memory for TimerState in C.")
            
        self.is_running = False

    def start(self):
        if not self.is_running:
            self.is_running = True
            timer_core_start(self.state)

    def stop(self):
        if self.is_running:
            self.is_running = False
            timer_core_stop(self.state)

    def join(self):
        # The stop function is now blocking and includes the join,
        # so this is mostly for API compatibility.
        pass

    def __dealloc__(self):
        # Ensure C memory is freed when the Python object is garbage collected
        if self.state is not NULL:
            timer_core_destroy(self.state)
