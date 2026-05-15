
import os
import threading
import time
from collections import deque

from .gpu_status import get_gpu_status

from .config import (
    PERF_HISTORY_SIZE,
    PER_THREAD_CPU_MONITOR_ENABLED,
    PER_THREAD_CPU_SAMPLE_SECONDS,
    PER_THREAD_CPU_TOP_N,
    PERF_SPIKE_LOG_THRESHOLD_PERCENT,
    PERF_SPIKE_LOG_FILE,
)

try:
    import psutil
except ImportError:
    psutil = None


_metrics = {
    "worker_left_ms": deque(maxlen=PERF_HISTORY_SIZE),
    "worker_right_ms": deque(maxlen=PERF_HISTORY_SIZE),
    "worker_point_ms": deque(maxlen=PERF_HISTORY_SIZE),
    "gui_frame_ms": deque(maxlen=PERF_HISTORY_SIZE),
    "queued_jobs": deque(maxlen=PERF_HISTORY_SIZE),
    "dropped_jobs": deque(maxlen=PERF_HISTORY_SIZE),
}
_counters = {
    "dropped_jobs": 0,
    "processed_jobs": 0,
}
_thread_cpu = []
_process_cpu_percent = 0.0
_last_spike_logged_at = 0.0
_monitor_started = False
_lock = threading.Lock()


def now():
    return time.perf_counter()


def record(name, value):
    with _lock:
        if name not in _metrics:
            _metrics[name] = deque(maxlen=PERF_HISTORY_SIZE)
        _metrics[name].append(float(value))


def increment(name, amount=1):
    with _lock:
        _counters[name] = _counters.get(name, 0) + amount


def average(name):
    with _lock:
        values = list(_metrics.get(name, []))
    if not values:
        return 0.0
    return sum(values) / len(values)


def latest(name):
    with _lock:
        values = _metrics.get(name)
        if not values:
            return 0.0
        return values[-1]


def _thread_name_map():
    return {
        getattr(thread, "native_id", None): thread.name
        for thread in threading.enumerate()
    }


def _write_spike_log(process_cpu, rows):
    global _last_spike_logged_at

    t = time.time()
    if t - _last_spike_logged_at < 5.0:
        return

    _last_spike_logged_at = t

    try:
        with open(PERF_SPIKE_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} process_cpu={process_cpu:.1f}%\n")
            for row in rows:
                f.write(
                    f"  {row.get('name','?')} tid={row.get('tid')} "
                    f"cpu={row.get('cpu_percent',0.0):.1f}% "
                    f"total={row.get('total_time',0.0):.2f}s\n"
                )
    except OSError:
        pass


def _monitor_loop():
    global _thread_cpu, _process_cpu_percent

    if psutil is None:
        return

    process = psutil.Process(os.getpid())
    previous = {}
    process.cpu_percent(None)

    while True:
        time.sleep(PER_THREAD_CPU_SAMPLE_SECONDS)

        try:
            process_cpu = process.cpu_percent(None)
            thread_names = _thread_name_map()
            current_threads = process.threads()
        except Exception:
            continue

        rows = []
        interval = max(PER_THREAD_CPU_SAMPLE_SECONDS, 0.001)
        cpu_count = max(os.cpu_count() or 1, 1)

        for thread_info in current_threads:
            total = float(thread_info.user_time + thread_info.system_time)
            last = previous.get(thread_info.id)
            previous[thread_info.id] = total

            if last is None:
                delta = 0.0
            else:
                delta = max(0.0, total - last)

            # psutil process CPU percent can exceed 100 on multicore systems.
            # This number is percent of one core for the thread.
            cpu_percent = (delta / interval) * 100.0

            rows.append({
                "tid": thread_info.id,
                "name": thread_names.get(thread_info.id, f"native-{thread_info.id}"),
                "cpu_percent": cpu_percent,
                "total_time": total,
            })

        rows.sort(key=lambda row: row["cpu_percent"], reverse=True)
        rows = rows[:PER_THREAD_CPU_TOP_N]

        with _lock:
            _thread_cpu = rows
            _process_cpu_percent = process_cpu

        if process_cpu >= PERF_SPIKE_LOG_THRESHOLD_PERCENT:
            _write_spike_log(process_cpu, rows)


def start_thread_cpu_monitor():
    global _monitor_started

    if _monitor_started:
        return

    if not PER_THREAD_CPU_MONITOR_ENABLED:
        return

    _monitor_started = True

    thread = threading.Thread(
        target=_monitor_loop,
        name="PerfMonitor",
        daemon=True,
    )
    thread.start()


def snapshot():
    with _lock:
        thread_cpu = list(_thread_cpu)
        process_cpu = _process_cpu_percent
        counters = dict(_counters)

    return {
        "worker_left_ms": average("worker_left_ms"),
        "worker_right_ms": average("worker_right_ms"),
        "worker_point_ms": average("worker_point_ms"),
        "gui_frame_ms": average("gui_frame_ms"),
        "queued_jobs": latest("queued_jobs"),
        "dropped_jobs": counters.get("dropped_jobs", 0),
        "processed_jobs": counters.get("processed_jobs", 0),
        "process_cpu_percent": process_cpu,
        "thread_cpu": thread_cpu,
        "psutil_available": psutil is not None,
        "cpu_note": "Percent is measured per logical CPU core; multicore process CPU can exceed 100%. native-* rows are non-Python/native library threads.",
        "gpu": get_gpu_status(),
    }
