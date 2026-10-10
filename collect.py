"""Collect the headline numbers of the six python-hpc repos into results/summary.json.

    ~/.venvs/pyhpc/bin/python collect.py

Every repo measured its own workload against its own ceiling on the same RTX 3070 Laptop GPU.
For each workload this script picks, from the repo's final ``results/rtx3070-laptop.json``:

* ``dsl``      best kernel written entirely in a Python GPU DSL (Triton, Warp, Numba-CUDA),
* ``compile``  best ``torch.compile`` route (the same tensor code, compiled by Inductor),
* ``cuda_c``   best hand-written CUDA C kernel (compiled by NVRTC, launched from Python),
* ``numpy``    the NumPy CPU reference,

and expresses each as a percentage of the workload's roofline (the field the repo itself uses),
at the largest problem size the group reached.
If a repo has ``results/counterparts.json`` (an established code run on the same GPU), its best
number per workload is added as ``counterpart``. Nothing is re-measured here.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITE = HERE.parent
DSL = ("triton", "warp", "numba")

Run = dict
Pick = Callable[[Run], bool]


def load(repo: str, name: str = "rtx3070-laptop.json") -> dict | None:
    p = SUITE / repo / "results" / name
    return json.loads(p.read_text()) if p.exists() else None


def commit(repo: str) -> str:
    r = subprocess.run(["git", "-C", str(SUITE / repo), "rev-parse", "--short", "HEAD"],
                       check=True, capture_output=True, text=True)
    return r.stdout.strip()


def size_of(r: Run) -> int:
    s = r.get("size") or [r.get("n", 0)]
    out = 1
    for x in s:
        out *= int(x)
    return out


def best(runs: list[Run], pick: Pick, pct: str) -> dict | None:
    """Best run of a group at the largest problem size that group reached.

    The repo papers base their claims on the largest grids; small grids can exceed the DRAM
    roofline because part of the working set stays in the 4 MB L2 cache.
    """
    c = [r for r in runs if pick(r) and r.get(pct) is not None]
    if not c:
        return None
    n = max(size_of(r) for r in c)
    r = max((r for r in c if size_of(r) == n), key=lambda r: r[pct])
    size = r.get("size") or [r.get("n")]
    return {"backend": r["backend"], "variant": r["variant"], "size": size,
            "value": r["median"], "metric": r.get("metric"), "pct": round(r[pct], 1)}


def groups(runs: list[Run], base: Pick, pct: str, dsl: Pick | None = None) -> dict:
    dsl = dsl or (lambda r: r["backend"] in DSL)
    return {
        "dsl": best(runs, lambda r: base(r) and dsl(r), pct),
        "compile": best(runs, lambda r: base(r) and r["backend"] == "torch-compile", pct),
        "cuda_c": best(runs, lambda r: base(r) and r["backend"] == "cupy-cuda-c", pct),
        "numpy": best(runs, lambda r: base(r) and r["backend"] == "numpy", pct),
    }


def workloads() -> list[dict]:
    out = []

    lbm = load("pylbm")["runs"]
    for lat, label in (("D2Q9", "LBM D2Q9"), ("D3Q19", "LBM D3Q19")):
        out.append({"repo": "pylbm", "workload": label, "bound": "memory",
                    "roofline": "copy bandwidth / minimal bytes per update",
                    **groups(lbm, lambda r, lat=lat: r["lattice"] == lat
                             and r["collision"] == "bgk" and r["storage"] == "f32",
                             "pct_copy_bw")})

    heat = load("pyheat3d")["runs"]
    for dims, label in ((2, "heat 2D"), (3, "heat 3D")):
        out.append({"repo": "pyheat3d", "workload": label, "bound": "memory",
                    "roofline": "copy bandwidth / 8 B per single-step update",
                    **groups(heat, lambda r, d=dims: r["dims"] == d and r["steps_per_pass"] == 1
                             and r["dtype"] == "float32", "pct_copy_bw")})

    grav = load("pygravity")["runs"]
    out.append({"repo": "pygravity", "workload": "N-body", "bound": "compute",
                "roofline": "measured FP32 FMA peak (20 flops per interaction)",
                **groups(grav, lambda r: r["precision"] == "fp32", "pct_peak_flops")})

    md = load("pymd")["runs"]
    out.append({"repo": "pymd", "workload": "LJ MD", "bound": "memory",
                "roofline": "copy bandwidth / minimal bytes per atom-step (lower bound)",
                **groups(md, lambda r: r["dtype"] in ("float32", "fp32", "mixed")
                         or "fp64" not in r["variant"], "pct_copy_bw")})

    sp = load("pyspectral")["runs"]
    for dim, label in ((2, "spectral 2D"), (3, "spectral 3D")):
        out.append({"repo": "pyspectral", "workload": label, "bound": "FFT",
                    "roofline": "cuFFT time for the step's transforms + pointwise traffic",
                    **groups(sp, lambda r, d=dim: r["dim"] == d and r["dtype"] == "float32"
                             and r["form"] == "basdevant", "pct_fft_roofline")})

    ising = load("pyising")["runs"]
    # the headline 1-bit multi-spin format (int8 kernels have a 8x lower roofline; the
    # tensor-op rungs exist only in int8, so they are quoted in that format)
    out.append({"repo": "pyising", "workload": "Ising 2D", "bound": "memory/RNG",
                "roofline": "min(copy bandwidth, Philox throughput) per spin update",
                **groups(ising, lambda r: r["dim"] == 2 and r["algorithm"] == "metropolis"
                         and (r["variant"] == "msc" or r["backend"] in ("torch-compile",
                                                                         "numpy")),
                         "pct_roofline")})
    return out


def _top(runs: list[Run], pick: Pick, key: str = "median", low: bool = False) -> Run | None:
    c = [r for r in runs if pick(r) and r.get(key) is not None]
    if not c:
        return None
    return (min if low else max)(c, key=lambda r: r[key])


def _cp(code: str, variant: str, pct: float, size: object, how: str, **extra) -> dict:
    return {"code": code, "variant": variant, "pct": round(pct, 1), "size": size,
            "conversion": how, **extra}


def counterparts(item: dict) -> list[dict]:
    """Established codes run on the same GPU, as % of the same roofline.

    Where the repo's counterpart runner applied the repo's own traffic model / ceiling to the
    counterpart, that percentage is used directly ("direct"). Otherwise the counterpart's
    throughput is divided by ours (best CUDA C, same session, same problem size) and multiplied
    by our CUDA C kernel's roofline fraction at that size from the main benchmark ("ratio").
    """
    repo, wl = item["repo"], item["workload"]
    doc = load(repo, "counterparts.json")
    if not doc or doc.get("provisional"):
        return []
    runs, out = doc["runs"], []
    if repo == "pylbm":
        lat = wl.split()[1]
        for code, var in (("FluidX3D", "FP32/FP32"), ("Lettuce", "native CUDA")):
            ok = [r for r in runs if r["code"] == code and r["variant"] == var
                  and r["lattice"] == lat]
            if ok:
                n = max(size_of(r) for r in ok)
                r = max((r for r in ok if size_of(r) == n), key=lambda r: r["pct_copy_bw"])
                out.append(_cp(code, var, r["pct_copy_bw"], r.get("size") or r.get("problem"),
                               "direct"))
    elif repo == "pyheat3d":
        d = 2 if "2D" in wl else 3
        ok = [r for r in runs if r["code"] == "pystencils" and r["dims"] == d]
        r = _top(ok, lambda r: True, "pct_copy_bw")
        if r:
            out.append(_cp("pystencils", r["variant"], r["pct_copy_bw"], r.get("size"), "direct"))
    elif repo == "pygravity":
        ok = [r for r in runs if r["code"] == "cuda-samples-nbody"]
        if ok:
            n = max(r["N"] for r in ok)
            r = max((r for r in ok if r["N"] == n), key=lambda r: r["pct_peak_flops"])
            out.append(_cp("CUDA Samples nbody", r["variant"], r["pct_peak_flops"], [n],
                           "direct"))
    elif repo == "pymd":
        main = [r for r in load("pymd")["runs"] if r["backend"] == "cupy-cuda-c"
                and r["variant"] == "full-fused"]
        for code, var in (("hoomd", "gpu-single"), ("lammps", "kokkos-single-half"),
                          ("lammps", "kokkos-single")):
            ok = [r for r in runs if r["code"] == code and r["variant"] == var
                  and "error" not in r]
            best_r = None
            for r in ok:
                ours = [o for o in runs if o["code"] == "pymd" and o["variant"] ==
                        "cupy-cuda-c/full-fused" and o["n"] == r["n"]]
                ref = [m for m in main if m["size"][0] == r["n"]]
                if ours and ref:
                    pct = ref[0]["pct_copy_bw"] * r["median"] / ours[0]["median"]
                    if best_r is None or pct > best_r["pct"]:
                        best_r = _cp({"hoomd": "HOOMD-blue", "lammps": "LAMMPS"}[code], var,
                                     pct, [r["n"]], "ratio", rate=r["median"],
                                     tuning_complete=r.get("tuning_complete"))
            if best_r:
                out.append(best_r)
    elif repo == "pyspectral" and "2D" in wl:
        main = {r["size"][0]: r for r in load("pyspectral")["runs"]
                if r["backend"] == "cupy-cuda-c" and r["variant"] == "fused-inplace"
                and r["dim"] == 2 and r["dtype"] == "float32" and r["form"] == "basdevant"}
        for var in ("if-rk4", "cn-rk4"):
            ok = [r for r in runs if r["code"] == "jax-cfd" and r["variant"] == var
                  and r["dtype"] == "float32"]
            if not ok:
                continue
            n = max(r["n"] for r in ok if r["n"] in main)
            cpr = next(r for r in ok if r["n"] == n)
            ours = [o for o in runs if o["code"] == "pyspectral" and o["variant"] ==
                    "cupy-cuda-c/fused-inplace" and o["n"] == n and o["dim"] == 2
                    and o["dtype"] == "float32"]
            if ours:
                pct = main[n]["pct_fft_roofline"] * ours[0]["step_ms"] / cpr["step_ms"]
                out.append(_cp("JAX-CFD", var, pct, [n], "ratio (time per step)"))
    elif repo == "pyising":
        main = {r["size"][0]: r for r in load("pyising")["runs"] if r["backend"] ==
                "cupy-cuda-c" and r["variant"] == "msc" and r["algorithm"] == "metropolis"}
        for var in ("cuIsingModel", "optimized_old"):
            ok = [r for r in runs if r["code"] == "NVIDIA/ising-gpu" and r["variant"] == var]
            best_r = None
            for r in ok:
                ours = [o for o in runs if o["code"] == "pyising" and o["variant"] ==
                        "cupy-cuda-c/msc" and o["L"] == r["L"]]
                if ours and r["L"] in main:
                    pct = main[r["L"]]["pct_roofline"] * r["median"] / ours[0]["median"]
                    if best_r is None or pct > best_r["pct"]:
                        best_r = _cp("NVIDIA ising-gpu", var, pct, [r["L"]], "ratio",
                                     rate=r["median"])
            if best_r:
                out.append(best_r)
    return out


def temporal_blocking() -> dict:
    """pyheat3d's temporal-blocking rung against AN5D (x the single-step roofline)."""
    doc = load("pyheat3d", "counterparts.json")
    out: dict = {}
    if not doc or doc.get("provisional"):
        return out
    for d in (2, 3):
        rs = [r for r in doc["runs"] if r["dims"] == d and r.get("steps_fused", 1) > 1]
        ours = [r for r in rs if r["code"] == "pyheat3d"]
        an5d = [r for r in rs if r["code"] == "AN5D"]
        if ours and an5d:
            o = max(ours, key=lambda r: r["median"])
            a = max(an5d, key=lambda r: r["median"])
            out[f"{d}D"] = {"ours": {"variant": o["variant"], "k": o["steps_fused"],
                                     "x_roof": o["x_single_step_roofline"]},
                            "an5d": {"variant": a["variant"], "k": a["steps_fused"],
                                     "x_roof": a["x_single_step_roofline"]}}
    return out


def main() -> None:
    w = workloads()
    for item in w:
        item["counterparts"] = counterparts(item)
    machine = load("pyising")["machine"]
    doc = {"machine": machine,
           "sources": {r: {"file": f"{r}/results/rtx3070-laptop.json", "commit": commit(r)}
                       for r in ("pylbm", "pyheat3d", "pygravity", "pymd", "pyspectral",
                                 "pyising")},
           "ceilings": {"copy_bandwidth_GBs": load("pyheat3d")["ceiling"]["copy_bandwidth_GBs"],
                        "spec_bandwidth_GBs": load("pyheat3d")["ceiling"]["spec_bandwidth_GBs"],
                        "fp32_fma_tflops": load("pygravity")["ceiling"]["fp32_tflops"],
                        "philox_calls_per_ns": load("pyising")["ceiling"]["philox_calls_per_ns"]},
           "workloads": w, "temporal_blocking": temporal_blocking()}
    (HERE / "results" / "summary.json").write_text(json.dumps(doc, indent=1) + "\n")
    for item in w:
        cells = [f"{k}={item[k]['pct']:5.1f}% ({item[k]['backend']}/{item[k]['variant']})"
                 for k in ("dsl", "compile", "cuda_c") if item.get(k)]
        cells += [f"{c['code']}/{c['variant']}={c['pct']:5.1f}%" for c in item["counterparts"]]
        print(f"{item['workload']:12s} " + "  ".join(cells))


if __name__ == "__main__":
    main()
