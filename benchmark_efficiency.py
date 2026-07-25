"""
Efficiency benchmarking.

What this script can measure ON WHATEVER MACHINE RUNS IT:
  - Parameter count, FLOPs (via thop), on-device latency (wall clock,
    warmed-up, averaged over N runs), peak GPU memory, and (if `pynvml` is
    available and a GPU is present) GPU utilization sampled during inference.

What it CANNOT do inside this sandbox, and what you need to run yourselves:
  - Actual Jetson / Raspberry Pi / Android latency and power-draw numbers.
    The `export_onnx()` function below produces a portable .onnx file per
    backbone; run it through TensorRT (Jetson) or TFLite (Android/RPi) on the
    real target hardware.
  - Power consumption requires a physical power meter (e.g. INA219 on a
    Jetson) or the board's onboard power-monitoring sysfs — no substitute
    exists in software alone.

Also adds MobileNetV3 / ShuffleNetV2 / SwinV2-Tiny as lightweight comparison
points per R2-#5, using the same benchmarking function as everything else so
numbers are apples-to-apples.
"""
import time
import numpy as np
import torch
import pandas as pd

from config import CFG
from models import build_model


def resolve_input_size(model, model_name, preferred=224):
    """Returns the input resolution to actually benchmark this model at.

    Prefers the pipeline's real deployment resolution (CFG.img_size=224) for
    every model, since that's what these architectures are actually trained
    and compared at in this project — using each model's own "native"
    ImageNet-pretraining resolution instead (e.g. 300 for EfficientNetV2-S,
    per its timm pretrained_cfg) would make the comparison inconsistent and
    unfair across the table, not more correct.

    Falls back to the model's timm pretrained_cfg input_size ONLY if a quick
    trial forward pass at `preferred` resolution fails — this is exactly
    what happens for swinv2_tiny_window8_256, whose patch-embed layer hard-
    asserts the input must match its pretraining resolution (256), unlike
    every other backbone in this pipeline which accepts arbitrary input
    sizes. The fallback resolution actually used is recorded in the output
    table so this deviation is visible, not silent."""
    try:
        device = next(model.parameters()).device
        with torch.no_grad():
            model(torch.randn(1, 3, preferred, preferred, device=device))
        return preferred
    except Exception:
        native = getattr(model, "backbone", model)
        cfg = getattr(native, "pretrained_cfg", None) or getattr(native, "default_cfg", {})
        native_size = cfg.get("input_size", (3, preferred, preferred))[-1]
        print(f"⚠️ {model_name} rejects {preferred}px input (structural constraint, "
              f"not a bug) — falling back to its native {native_size}px for this "
              f"model only. This is recorded in the input_size column below.")
        return native_size


def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def estimate_flops(model, input_size=(1, 3, 224, 224), device="cuda"):
    try:
        from thop import profile
        dummy = torch.randn(input_size).to(device)
        flops, _ = profile(model.to(device), inputs=(dummy,), verbose=False)
        return flops / 1e9
    except Exception as e:
        print(f"⚠️ FLOPs estimation failed for this backbone: {e}")
        return float("nan")


@torch.no_grad()
def measure_latency(model, device, input_size=(1, 3, 224, 224), n_warmup=20, n_runs=100):
    model = model.to(device).eval()
    dummy = torch.randn(input_size).to(device)
    for _ in range(n_warmup):
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "latency_ms_mean": float(np.mean(times)),
        "latency_ms_std": float(np.std(times)),
        "latency_ms_p95": float(np.percentile(times, 95)),
    }


def reset_gpu_memory_stats(device):
    """Call BEFORE running inference, so the peak captured afterward reflects
    what actually happened during that run."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_gpu_memory_mb(device):
    """Call AFTER running inference. Reading immediately after a reset (the
    previous version of this function did both in one call, back to back,
    with no inference in between) always returns ~0 — the reset wipes out
    the peak before it can be read, so it never reflected real memory use."""
    if device.type != "cuda":
        return float("nan")
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def export_onnx(model, out_path, input_size=(1, 3, 224, 224), device="cpu"):
    model = model.to(device).eval()
    dummy = torch.randn(input_size).to(device)
    torch.onnx.export(model, dummy, out_path, input_names=["input"], output_names=["logits"],
                       dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
    print(f"✅ Exported {out_path} — benchmark this on the actual target device "
          f"(Jetson/RPi/Android) for real latency + power numbers.")


def benchmark_all(num_classes, cfg=CFG, include_lightweight=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = list(cfg.backbones) + (list(cfg.lightweight_baselines) if include_lightweight else [])
    rows = []
    csv_path = f"{cfg.output_dir}/efficiency_benchmark.csv"
    for name in names:
        try:
            model = build_model(name, num_classes).to(device).eval()
            res = resolve_input_size(model, name, preferred=cfg.img_size)
            input_size = (1, 3, res, res)

            params = count_parameters(model)
            flops = estimate_flops(model, input_size=input_size, device=device)
            reset_gpu_memory_stats(device)
            lat = measure_latency(model, device, input_size=input_size)
            mem = peak_gpu_memory_mb(device)
            rows.append({"model": name, "input_size_px": res, "params_M": params,
                         "flops_G": flops, "gpu_mem_MB": mem, **lat})

            # ONNX export is a secondary output (only useful once you have
            # actual Jetson/RPi/Android hardware to run it on) — a failure
            # here must not throw away metrics already computed above.
            try:
                export_onnx(model, f"{cfg.output_dir}/{name}.onnx", input_size=input_size, device="cpu")
            except Exception as e:
                print(f"⚠️ ONNX export failed for {name} (metrics above are still "
                      f"valid and saved): {e}")

        except Exception as e:
            # Nothing about ANY single model — a resolution mismatch, an
            # OOM, an unsupported op, anything — should be able to take
            # down every model queued after it. This is exactly what
            # happened when swinv2_tiny_window8_256's hardcoded input-size
            # assertion crashed the whole script with 2 of 6 models still
            # unbenchmarked, and no CSV had been written yet for them.
            print(f"⚠️ Benchmarking failed entirely for {name}, skipping it "
                  f"(other models are unaffected): {e}")
            continue

        # Write after every model, not just at the end, so a later failure
        # can't erase earlier results.
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    CFG.ensure_dirs()
    df = benchmark_all(len(CFG.class_names))
    print(df)
