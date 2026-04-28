#ifndef TIMER_CORE_H
#define TIMER_CORE_H

#include <stdint.h>
#include <stdbool.h>
#include <windows.h>

// Define a C function pointer type for our trigger function.
// It matches the signature of SC2_Cam.PCO_ForceTrigger
// (takes HANDLE, uint16_t, returns an int).
typedef int (__stdcall *trigger_func_ptr_t)(HANDLE, uint16_t*);

// C结构，用于保存一个定时器实例的所有状态
typedef struct {
    // 定时参数，单位为 QPC "ticks"
    LONGLONG interval_ticks;
    LONGLONG busy_wait_ticks;

    // 用户传入的相机句柄和回调函数指针
    HANDLE hCamera; // PCO 相机使用HANDLE类型的句柄
    trigger_func_ptr_t trigger_func;

    // 线程和控制标志
    HANDLE hThread;
    volatile int stop_flag; // volatile至关重要，防止编译器过度优化
    int priority;           // 线程优先级

    // Windows特有的计时器频率
    LARGE_INTEGER performance_frequency;
    
    // 是否打印调试信息
    int debug_print; // 0: 不打印, 1: 打印
} TimerState;

// Functions exported by our C module
TimerState* timer_core_create(
    uint64_t interval_ns,
    trigger_func_ptr_t trigger_func,
    HANDLE hCamera,
    int busy_wait_ns,
    int priority,
    int debug_print
);

void timer_core_start(TimerState* state);
void timer_core_stop(TimerState* state);
void timer_core_destroy(TimerState* state);

#endif // TIMER_CORE_H
