import os

# GPU-resource tests use synthetic inventories; production still requires the
# operator to set WOV2_GPU_ALLOWLIST explicitly.
os.environ.setdefault("WOV2_GPU_ALLOWLIST", "0,1,2,3,4,5,6")
