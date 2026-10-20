import os
import sys

if sys.platform.startswith("linux"):
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
os.environ.setdefault("MIOPEN_LOG_LEVEL", "3")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")

import torch

_CPU_FALLBACK_USED = False


def is_rocm():
    return getattr(torch.version, "hip", None) is not None


def pick_device():
    forced = os.environ.get("AUTOMATA_DEVICE", "").strip().lower()
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def use_half(device):
    if device.type != "cuda":
        return False
    return os.environ.get("AUTOMATA_FP16", "1") != "0"


def describe(device):
    if device.type != "cuda":
        return f"{device} (torch {torch.__version__})"
    backend = f"ROCm/HIP {torch.version.hip}" if is_rocm() else f"CUDA {torch.version.cuda}"
    name = torch.cuda.get_device_name(device)
    arch = getattr(torch.cuda.get_device_properties(device), "gcnArchName", "")
    return f"{device} [{name} {arch}] via {backend}, torch {torch.__version__}"


def load_birefnet(model_id="ZhengPeng7/BiRefNet"):
    from transformers import AutoModelForImageSegmentation
    device = pick_device()
    model = AutoModelForImageSegmentation.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    half = use_half(device)
    try:
        model.to(device)
        if half:
            model.half()
    except RuntimeError as e:
        print(f"  [gpu] moving model to {device} failed ({e}); using CPU")
        device, half = torch.device("cpu"), False
        model.float().to(device)
    print(f"  [gpu] BiRefNet on {describe(device)}{' fp16' if half else ''}")
    return model, device


def birefnet_forward(model, tensor):
    global _CPU_FALLBACK_USED
    p = next(model.parameters())
    try:
        with torch.inference_mode():
            preds = model(tensor.to(p.device, p.dtype))
            if isinstance(preds, (list, tuple)):
                preds = preds[-1]
            return torch.sigmoid(preds.float())
    except RuntimeError as e:
        if p.device.type == "cpu" or os.environ.get("AUTOMATA_NO_FALLBACK") == "1":
            raise
        print(f"  [gpu] forward failed on {p.device} ({str(e).splitlines()[0]}); retrying on CPU")
        _CPU_FALLBACK_USED = True
        model.float().to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        with torch.inference_mode():
            preds = model(tensor.float().cpu())
            if isinstance(preds, (list, tuple)):
                preds = preds[-1]
            return torch.sigmoid(preds.float())


def _smoke_test():
    import time
    dev = pick_device()
    print(f"torch {torch.__version__}, hip={torch.version.hip}, cuda={torch.version.cuda}")
    print(f"cuda.is_available() = {torch.cuda.is_available()}")
    print(f"device: {describe(dev)}")
    if dev.type != "cuda":
        print("No GPU visible to torch. Check the install (see ROCM_SETUP.md).")
        return 1
    a = torch.randn(2048, 2048, device=dev)
    b = (a @ a).sum().item()
    x = torch.randn(1, 3, 256, 256, device=dev)
    conv = torch.nn.Conv2d(3, 16, 3, padding=1).to(dev)
    conv(x).sum().item()
    print(f"matmul + conv OK ({b:.1f})")
    if "--birefnet" in sys.argv:
        model, dev = load_birefnet()
        t = torch.randn(1, 3, 1024, 1024)
        for i in range(3):
            t0 = time.time()
            out = birefnet_forward(model, t)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print(f"  BiRefNet pass {i + 1}: {time.time() - t0:.2f}s, out {tuple(out.shape)} on {out.device}")
        if _CPU_FALLBACK_USED:
            print("WARNING: GPU forward failed, CPU fallback was used. Try AUTOMATA_FP16=0.")
            return 2
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(_smoke_test())
