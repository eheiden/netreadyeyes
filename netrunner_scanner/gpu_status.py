
import os

_provider_info = {
    "available": [],
    "active": "unknown",
    "gpu_available": False,
    "gpu_enabled": True,
    "note": "",
    "last_benchmark_ms": None,
}


def _detect_available_providers():
    try:
        import onnxruntime as ort
        return list(ort.get_available_providers())
    except Exception:
        return []


def configure_gpu(enabled=True):
    """Record GPU preference and expose provider diagnostics.

    CollectorVision owns the ONNX sessions internally, so this module cannot
    guarantee a provider switch after models have already been loaded. The
    preference is set early from netreadyeyes.py and shown in the UI so I can
    tell whether CUDA/TensorRT is available on the machine.
    """
    enabled = bool(enabled)
    available = _detect_available_providers()
    gpu_providers = [
        provider for provider in available
        if provider in ("TensorrtExecutionProvider", "CUDAExecutionProvider", "DmlExecutionProvider")
    ]

    if enabled and gpu_providers:
        active = gpu_providers[0]
        os.environ.setdefault("NET_READY_EYES_GPU_ENABLED", "1")
        note = "GPU provider available"
    elif enabled:
        active = "CPUExecutionProvider" if "CPUExecutionProvider" in available else "unknown"
        note = "GPU requested, but no GPU ONNX provider is available"
    else:
        active = "CPUExecutionProvider" if "CPUExecutionProvider" in available else "unknown"
        os.environ["NET_READY_EYES_GPU_ENABLED"] = "0"
        note = "GPU disabled in Net Ready Eyes settings"

    _provider_info.update({
        "available": available,
        "active": active,
        "gpu_available": bool(gpu_providers),
        "gpu_enabled": enabled,
        "note": note,
    })
    return dict(_provider_info)


def get_gpu_status():
    if not _provider_info["available"]:
        configure_gpu(_provider_info.get("gpu_enabled", True))
    return dict(_provider_info)
