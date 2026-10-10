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


def main() -> None:
    w = workloads()
    for item in w:
        cp = load(item["repo"], "counterparts.json")
        item["counterpart"] = None
        if cp:
            item["counterpart_source"] = f"{item['repo']}/results/counterparts.json"
    machine = load("pyising")["machine"]
    doc = {"machine": machine,
           "sources": {r: {"file": f"{r}/results/rtx3070-laptop.json", "commit": commit(r)}
                       for r in ("pylbm", "pyheat3d", "pygravity", "pymd", "pyspectral",
                                 "pyising")},
           "workloads": w}
    (HERE / "results" / "summary.json").write_text(json.dumps(doc, indent=1) + "\n")
    for item in w:
        cells = [f"{k}={item[k]['pct']:5.1f}% ({item[k]['backend']}/{item[k]['variant']})"
                 for k in ("dsl", "compile", "cuda_c") if item.get(k)]
        print(f"{item['workload']:12s} " + "  ".join(cells))


if __name__ == "__main__":
    main()
