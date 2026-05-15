# Keep native libraries from creating a pile of helper threads. This needs to
# happen before collector_vision / numpy / OpenCV do much work.
import os

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "1")

# onnxruntime / OpenCV may still make native worker threads, but this helps.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
