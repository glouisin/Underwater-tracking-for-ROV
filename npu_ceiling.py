#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# npu_ceiling.py — plafond matériel d'un modèle, isolé du pipeline.
#
# Mesure :
#   - latence par invoke (p50/p90/p95/p99) hors caméra / GStreamer / dessin
#   - débit maximal atteignable  -> dénominateur du duty cycle NPU
#   - coût set_tensor / get_tensor (copies CPU<->DSP) vs invoke pur
#   - balayage de htp_performance_mode
#   - dérive thermique sur rafale longue
#   - LOGS BACKEND QNN capturés au niveau fd : temps par étage de préparation
#     du graphe, allocation VTCM, et surtout le "DDR bandwidth summary"
#     (spill/fill = débordement VTCM, octets lus/écrits par graphe).
#
# ATTENTION : toutes les options du delegate QNN sont parsées par stoi().
# Une valeur non numérique provoque un abort C++ NON RATTRAPABLE depuis
# Python (std::invalid_argument -> terminate). D'où la validation en amont.
#
# Usage :
#   python3 npu_ceiling.py --model M.tflite --sweep-perf --n 300
#   python3 npu_ceiling.py --model M.tflite --n 200 --qnn-profiling 2
#   python3 npu_ceiling.py --model M.tflite --cpu
#   python3 npu_ceiling.py --model M.tflite --n 5000 --thermal
# -----------------------------------------------------------------------------
import os

QAIRT = os.environ.get("QAIRT", "/opt/qcom/qairt-new/qairt/2.48.0.260626")
os.environ["ADSP_LIBRARY_PATH"] = (
    f"{QAIRT}/lib/hexagon-v73/unsigned;"
    "/opt/qcom/qirp-sdk/lib/aarch64-oe-linux-gcc11.2;"
    "/opt/qcom/qirp-sdk/lib/hexagon-v73/unsigned;"
    + os.environ.get("ADSP_LIBRARY_PATH", "")
)

import re
import sys
import glob
import json
import time
import tempfile
import argparse

import numpy as np
import ai_edge_litert.interpreter as tflite

DELEGATE = f"{QAIRT}/lib/aarch64-oe-linux-gcc11.2/libQnnTFLiteDelegate.so"


# --------------------------------------------------------------------------
# Capture des logs backend (émis par la couche C++, invisibles pour sys.stdout)
# --------------------------------------------------------------------------
class CaptureFD:
    def __init__(self):
        self.path = tempfile.mktemp(suffix=".qnnlog")
        self.text = ""

    def __enter__(self):
        sys.stdout.flush()
        sys.stderr.flush()
        self.saved = (os.dup(1), os.dup(2))
        self.fh = open(self.path, "w+b")
        os.dup2(self.fh.fileno(), 1)
        os.dup2(self.fh.fileno(), 2)
        return self

    def __exit__(self, *exc):
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(self.saved[0], 1)
        os.dup2(self.saved[1], 2)
        os.close(self.saved[0])
        os.close(self.saved[1])
        self.fh.flush()
        self.fh.seek(0)
        self.text = self.fh.read().decode(errors="ignore")
        self.fh.close()
        try:
            os.unlink(self.path)
        except OSError:
            pass
        return False


def parse_qnn_log(txt):
    """Extrait les étages de préparation et le résumé de bande passante DDR."""
    out = {"stages_us": {}, "ddr": {}}
    for name, us in re.findall(r"Completed stage: (.+?) \((\d+) us\)", txt):
        out["stages_us"][name.strip()] = int(us)
    for k in ("spill_bytes", "fill_bytes", "write_total_bytes", "read_total_bytes"):
        m = re.search(rf"{k}=(\d+)", txt)
        if m:
            out["ddr"][k] = int(m.group(1))
    if out["stages_us"]:
        out["prepare_total_ms"] = round(sum(out["stages_us"].values()) / 1000.0, 1)
    # lignes de profiling d'exécution éventuelles (profiling=1|2)
    ex = re.findall(r"(?:Execute|execute|Graph Execute)[^\n]*?(\d+)\s*us", txt)
    if ex:
        v = [int(x) for x in ex]
        out["execute_us"] = {"n": len(v), "mean": round(float(np.mean(v)), 1),
                             "min": min(v), "max": max(v)}
    return out


# --------------------------------------------------------------------------
def thermals():
    out = {}
    for d in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        try:
            t = open(os.path.join(d, "type")).read().strip()
            v = int(open(os.path.join(d, "temp")).read()) / 1000.0
            if v > 5:
                out[t] = round(v, 1)
        except Exception:
            pass
    return out


def npu_temp(th):
    v = [x for k, x in th.items() if "nspss" in k]
    return max(v) if v else None


def pct(v, p):
    return float(np.percentile(np.asarray(v, float), p))


def build(model, use_htp, perf_mode, profiling=None):
    if not use_htp:
        return tflite.Interpreter(model_path=model), None
    opts = {"backend_type": "htp", "htp_performance_mode": str(perf_mode)}
    if profiling:
        # Nom confirmé par `strings libQnnTFLiteDelegate.so` (QAIRT 2.48.0) :
        # la clé est "profiling" (et non "profiling_level" comme qnn-net-run).
        opts["profiling"] = str(profiling)
    # GARDE-FOU : stoi() sur une valeur non numérique = abort C++ non rattrapable.
    for k, v in opts.items():
        if k != "backend_type" and not str(v).lstrip("-").isdigit():
            raise ValueError(f"option '{k}={v}' non numérique -> abort C++ garanti")
    dl = tflite.load_delegate(DELEGATE, options=opts)
    return tflite.Interpreter(model_path=model, experimental_delegates=[dl]), opts


def run(model, n, use_htp, perf_mode, warmup, thermal, profiling, quiet):
    cap = CaptureFD()
    t_load0 = time.perf_counter()
    with cap:
        itp, opts = build(model, use_htp, perf_mode, profiling)
        itp.allocate_tensors()
    t_load = (time.perf_counter() - t_load0) * 1000
    qnn_prepare = parse_qnn_log(cap.text)
    if not quiet and cap.text.strip():
        print(f"  [qnn] prepare {qnn_prepare.get('prepare_total_ms', '?')} ms, "
              f"ddr={qnn_prepare.get('ddr')}")

    ind, outd = itp.get_input_details()[0], itp.get_output_details()[0]
    shape, dtype = ind["shape"], ind["dtype"]
    rng = np.random.default_rng(0)
    if dtype in (np.uint8, np.int8):
        lo, hi = (0, 255) if dtype == np.uint8 else (-128, 127)
        tensor = rng.integers(lo, hi, size=shape, dtype=dtype)
    else:
        tensor = rng.standard_normal(shape).astype(dtype)

    for _ in range(warmup):
        itp.set_tensor(ind["index"], tensor)
        itp.invoke()

    lat, t_set, t_get, temps = [], [], [], []
    th0 = thermals()
    cap2 = CaptureFD()
    t0 = time.perf_counter()
    with cap2:
        for i in range(n):
            a = time.perf_counter()
            itp.set_tensor(ind["index"], tensor)
            b = time.perf_counter()
            itp.invoke()
            c = time.perf_counter()
            itp.get_tensor(outd["index"])
            d = time.perf_counter()
            t_set.append((b - a) * 1000)
            lat.append((c - b) * 1000)
            t_get.append((d - c) * 1000)
            if thermal and i % 50 == 0:
                th = thermals()
                temps.append({"i": i, "t": round(time.perf_counter() - t0, 2),
                              "npu": npu_temp(th), "all": th,
                              "lat_ms": round(lat[-1], 2)})
    wall = time.perf_counter() - t0
    th1 = thermals()
    qnn_exec = parse_qnn_log(cap2.text)

    return {
        "model": os.path.basename(model),
        "backend": "htp" if use_htp else "cpu",
        "htp_performance_mode": perf_mode if use_htp else None,
        "profiling": profiling,
        "delegate_options": opts,
        "input_shape": [int(x) for x in shape],
        "input_dtype": str(dtype),
        "load_alloc_ms": round(t_load, 1),
        "n": n,
        "wall_s": round(wall, 2),
        "ceiling_inf_per_s": round(n / wall, 2),
        "invoke_ms": {f"p{p}": round(pct(lat, p), 3) for p in (50, 90, 95, 99)} | {
            "mean": round(float(np.mean(lat)), 3),
            "min": round(min(lat), 3), "max": round(max(lat), 3),
            "std": round(float(np.std(lat)), 3)},
        "set_tensor_ms_mean": round(float(np.mean(t_set)), 3),
        "get_tensor_ms_mean": round(float(np.mean(t_get)), 3),
        "first10_vs_last10_ms": [round(float(np.mean(lat[:10])), 2),
                                 round(float(np.mean(lat[-10:])), 2)],
        "npu_temp_start_C": npu_temp(th0),
        "npu_temp_end_C": npu_temp(th1),
        "qnn_prepare": qnn_prepare,
        "qnn_execute": qnn_exec if qnn_exec.get("execute_us") else None,
        "qnn_execute_log_head": cap2.text[:2000] if cap2.text.strip() else None,
        "thermal_trace": temps if thermal else None,
    }


def table(results):
    print(f"\n{'backend':<8} {'perf':<5} {'p50':>8} {'p95':>8} {'p99':>8} "
          f"{'inf/s':>8} {'set':>6} {'get':>6} {'prep_ms':>8} {'Tnpu':>12}")
    print("-" * 92)
    for r in results:
        iv = r["invoke_ms"]
        t = (f"{r['npu_temp_start_C']}->{r['npu_temp_end_C']}"
             if r["npu_temp_start_C"] else "n/a")
        print(f"{r['backend']:<8} {str(r['htp_performance_mode']):<5} "
              f"{iv['p50']:>8.2f} {iv['p95']:>8.2f} {iv['p99']:>8.2f} "
              f"{r['ceiling_inf_per_s']:>8.1f} {r['set_tensor_ms_mean']:>6.2f} "
              f"{r['get_tensor_ms_mean']:>6.2f} "
              f"{str(r['qnn_prepare'].get('prepare_total_ms', '-')):>8} {t:>12}")
    ddr = next((r["qnn_prepare"].get("ddr") for r in results
                if r["qnn_prepare"].get("ddr")), None)
    if ddr:
        print(f"\nDDR / graphe : read={ddr.get('read_total_bytes',0)/1e6:.2f} Mo  "
              f"write={ddr.get('write_total_bytes',0)/1e6:.2f} Mo  "
              f"spill={ddr.get('spill_bytes')}  fill={ddr.get('fill_bytes')}"
              + ("   (spill=fill=0 : le graphe tient en VTCM)"
                 if ddr.get("spill_bytes") == 0 and ddr.get("fill_bytes") == 0 else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--cpu", action="store_true", help="référence CPU (sans delegate)")
    ap.add_argument("--perf", default="1", help="htp_performance_mode (entier)")
    ap.add_argument("--sweep-perf", action="store_true", help="balaye les modes 0..4")
    ap.add_argument("--thermal", action="store_true")
    ap.add_argument("--qnn-profiling", default=None, choices=["1", "2"],
                    help="option 'profiling' du delegate : 1=basic, 2=detailed")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out = a.out or f"ceiling_{os.path.basename(a.model).replace('.tflite','')}.json"
    results = []

    def save():   # écriture incrémentale : un crash ne perd plus tout le run
        with open(out, "w") as f:
            json.dump(results, f, indent=2)

    modes = ["0", "1", "2", "3", "4"] if a.sweep_perf else ([] if a.cpu else [a.perf])
    if a.cpu:
        try:
            results.append(run(a.model, a.n, False, None, a.warmup, a.thermal,
                               None, a.quiet)); save()
        except Exception as e:
            print(f"  cpu: FAILED ({e})")
    for m in modes:
        print(f"--- htp_performance_mode={m} ---")
        try:
            results.append(run(a.model, a.n, True, m, a.warmup, a.thermal,
                               a.qnn_profiling, a.quiet)); save()
        except Exception as e:
            print(f"  perf_mode={m}: FAILED ({e})")

    if results:
        table(results)
    save()
    print(f"\n-> {out}")
    print("ceiling_inf_per_s est le dénominateur du duty cycle NPU.")


if __name__ == "__main__":
    main()