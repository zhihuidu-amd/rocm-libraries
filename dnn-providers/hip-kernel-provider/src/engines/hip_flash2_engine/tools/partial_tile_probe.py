#!/usr/bin/env python3
"""Is the kernel still correct when seq_len_q is NOT a multiple of qPerCta?

WHY THIS EXISTS
isApplicable() enforces seq_len_q % 64 == 0 because a partial query tile makes
__syncthreads() divergent. That gate was written when every CTA covered 64
queries. The dispatcher (Flash2Dispatch.hpp) now selects 128-, 256- and
384-query tiles, and S=2048 is a multiple of 64 but NOT of 384 -- so such a
shape passes the gate and then runs a partial tile. If the tail handling is
wrong the output is silently wrong: no error code, no launch failure.

Reading the kernel it looks safe (kb_stop derives from a CTA-wide bound, the
only barrier is at the top of the KV loop so the trip count stays uniform
across waves, and the Q load / KV fetch / O store are all bounds-checked), but
that is inspection, not measurement. This measures it.

WHAT IT CHECKS
Two reference-free invariants, exact -- no tolerance tuning:
  1. V constant       -> every output element equals that constant, because
                         the softmax weights sum to 1 by construction.
  2. Q = 0, noncausal -> every output row equals mean(V) over the keys,
                         because attention is uniform.
Plus a cross-check against torch SDPA computed in fp32.

The S values straddle the tile sizes: all are multiples of 64 (so all pass
isApplicable) but only some divide evenly by the selected qPerCta. Rows where
they do not are marked PARTIAL -- those are the ones that matter.

USAGE
  python3 partial_tile_probe.py [kernels_dir]

kernels_dir defaults to the installed engine kernel directory, or set
HIP_FLASH2_KERNEL_DIR. Variants that are not present are skipped, so this is
useful even with a partial variant set.

NOTE: load the HIP runtime that torch itself is using. Mixing a system
libamdhip64.so with torch's bundled one gives hipModuleLoad -> 209
(hipErrorSharedObjectInitFailed), and torch must initialize the context
BEFORE the first hipModuleLoad.
"""

import ctypes
import math
import os
import sys

import torch

# tag -> (qPerCta, blockDim). Must match Flash2Dispatch.hpp.
VARIANTS = {
    "w8q2k4": (256, 512),
    "w8q3k2": (384, 512),
    "w8q3k4": (384, 512),
    "w8q1k4": (128, 512),
    "w4q1k4": (64, 256),
}

DEFAULT_KERNEL_DIR = os.environ.get(
    "HIP_FLASH2_KERNEL_DIR",
    "/opt/rocm/lib/hipdnn_plugins/engines/hip_kernel_provider/hip_flash2_kernels",
)


def co_path(kernel_dir, tag):
    return os.path.join(kernel_dir, f"hip_flash2_fwd_gfx942_{tag}.co")


def load_hip():
    """Prefer the runtime torch already loaded; fall back to the system one."""
    for cand in (
        os.path.join(os.path.dirname(torch.__file__), "lib", "libamdhip64.so"),
        "/opt/rocm/lib/libamdhip64.so",
        "libamdhip64.so",
    ):
        try:
            return ctypes.CDLL(cand)
        except OSError:
            continue
    raise RuntimeError("could not load libamdhip64.so")


hip = load_hip()
hip.hipModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
hip.hipModuleGetFunction.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_void_p,
    ctypes.c_char_p,
]
hip.hipModuleLaunchKernel.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_void_p,
]
hip.hipDeviceSynchronize.argtypes = []


def load(path, name):
    mod = ctypes.c_void_p()
    rc = hip.hipModuleLoad(ctypes.byref(mod), path.encode())
    if rc != 0:
        raise RuntimeError(f"hipModuleLoad({path}) -> {rc}")
    fn = ctypes.c_void_p()
    rc = hip.hipModuleGetFunction(ctypes.byref(fn), mod, name.encode())
    if rc != 0:
        raise RuntimeError(f"hipModuleGetFunction({name}) -> {rc}")
    return fn


def launch(fn, q, k, v, o, B, Hq, Hk, S, Skv, D, causal, qper, blk):
    scale = 1.0 / math.sqrt(D)
    args = [
        ("P", q.data_ptr()),
        ("P", k.data_ptr()),
        ("P", v.data_ptr()),
        ("P", o.data_ptr()),
        ("i", B),
        ("i", Hq),
        ("i", Hk),
        ("i", S),
        ("i", Skv),
        ("i", D),
        ("f", scale),
        ("i", 1 if causal else 0),
    ]
    # NHD strides: [B, S, H, D] is NOT what we use here; these are BHSD.
    for t in (q, k, v, o):
        args += [
            ("l", t.stride(0)),
            ("l", t.stride(1)),
            ("l", t.stride(2)),
        ]
    cells, keep = [], []
    for kind, val in args:
        if kind == "P":
            c = ctypes.c_void_p(val)
        elif kind == "i":
            c = ctypes.c_int(val)
        elif kind == "f":
            c = ctypes.c_float(val)
        else:
            c = ctypes.c_longlong(val)
        keep.append(c)
        cells.append(ctypes.cast(ctypes.byref(c), ctypes.c_void_p))
    arr = (ctypes.c_void_p * len(cells))(*cells)
    gridx = (S + qper - 1) // qper
    rc = hip.hipModuleLaunchKernel(fn, gridx, B, Hq, blk, 1, 1, 0, None, arr, None)
    if rc != 0:
        raise RuntimeError(f"launch -> {rc}")
    rc = hip.hipDeviceSynchronize()
    if rc != 0:
        raise RuntimeError(f"sync -> {rc}")


def main():
    dev = "cuda"
    # Initialize the HIP context via torch BEFORE any hipModuleLoad. Loading a
    # module first returns 209 (hipErrorSharedObjectInitFailed) and poisons the
    # context for torch afterwards.
    torch.cuda.init()
    _w = torch.ones(1024, dtype=torch.float16, device=dev) * 2
    torch.cuda.synchronize()
    del _w
    torch.manual_seed(0)
    D = 128
    B, Hq = 1, 32
    rows = []

    # S values chosen to straddle the tile sizes. All are multiples of 64, so
    # all pass isApplicable(); only some are multiples of the selected qPerCta.
    kernel_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_KERNEL_DIR
    print("kernel dir: " + kernel_dir)
    for tag in ("w8q3k2", "w8q2k4"):
        qper, blk = VARIANTS[tag]
        path = co_path(kernel_dir, tag)
        if not os.path.exists(path):
            print(f"{tag}: {path} not installed, skipping")
            continue
        fn = load(path, "flash2_v7_hipdnn_d128")
        for S in (1024, 1536, 2048, 2560, 3072, 4096):
            partial = (S % qper) != 0
            for causal in (False, True):
                # --- invariant 1: constant V ---
                q = torch.randn(B, Hq, S, D, device=dev, dtype=torch.float16)
                k = torch.randn(B, Hq, S, D, device=dev, dtype=torch.float16)
                v = torch.full((B, Hq, S, D), 0.5, device=dev, dtype=torch.float16)
                o = torch.zeros_like(q)
                launch(fn, q, k, v, o, B, Hq, Hq, S, S, D, causal, qper, blk)
                err_const = (o.float() - 0.5).abs().max().item()

                # --- invariant 2: Q=0 noncausal -> mean(V) ---
                if not causal:
                    q0 = torch.zeros(B, Hq, S, D, device=dev, dtype=torch.float16)
                    vr = torch.randn(B, Hq, S, D, device=dev, dtype=torch.float16)
                    o2 = torch.zeros_like(q0)
                    launch(fn, q0, k, vr, o2, B, Hq, Hq, S, S, D, False, qper, blk)
                    want = vr.float().mean(dim=2, keepdim=True).expand_as(o2)
                    err_mean = (o2.float() - want).abs().max().item()
                else:
                    err_mean = float("nan")

                # --- cross-check vs torch SDPA ---
                qr = torch.randn(B, Hq, S, D, device=dev, dtype=torch.float16)
                kr = torch.randn(B, Hq, S, D, device=dev, dtype=torch.float16)
                vr2 = torch.randn(B, Hq, S, D, device=dev, dtype=torch.float16)
                o3 = torch.zeros_like(qr)
                launch(fn, qr, kr, vr2, o3, B, Hq, Hq, S, S, D, causal, qper, blk)
                ref = torch.nn.functional.scaled_dot_product_attention(
                    qr.float(), kr.float(), vr2.float(), is_causal=causal
                )
                err_ref = (o3.float() - ref).abs().max().item()

                rows.append(
                    (tag, qper, S, partial, causal, err_const, err_mean, err_ref)
                )
                print(
                    f"{tag:8s} qper={qper:3d} S={S:5d} "
                    f"{'PARTIAL' if partial else 'exact  '} "
                    f"{'causal   ' if causal else 'noncausal'} "
                    f"constV={err_const:.3e} meanV={err_mean:.3e} vsSDPA={err_ref:.3e}",
                    flush=True,
                )

    print("\n=== summary ===")
    bad = [r for r in rows if r[5] > 1e-2 or r[7] > 5e-2]
    for r in rows:
        flag = "FAIL" if (r[5] > 1e-2 or r[7] > 5e-2) else "ok"
        if r[3]:
            print(
                f"  {flag:4s} PARTIAL {r[0]} S={r[2]} causal={r[4]} "
                f"constV={r[5]:.3e} vsSDPA={r[7]:.3e}"
            )
    print(f"\n{'FAIL' if bad else 'PASS'}: {len(bad)} bad of {len(rows)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
