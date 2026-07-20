#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# power_probe.py — décide si les canaux iio du PMIC sont exploitables pour
# mesurer l'énergie par inférence, ou s'il faut un wattmètre externe.
#
# Protocole : idle / charge / idle, échantillonné à 2 Hz, avec test de
# significativité (le delta doit dépasser 3 sigma du bruit au repos).
# Le test à un seul point du probe précédent ne pouvait pas conclure.
#
#   python3 power_probe.py --load cpu               # charge CPU 6 threads
#   python3 power_probe.py --load npu --model X.tflite   # charge NPU réelle
#   python3 power_probe.py --load none --dur 120    # bruit de fond seul
#
# Sortie : power_probe_<load>.csv + verdict sur stdout.
# -----------------------------------------------------------------------------
import os
import csv
import glob
import time
import argparse
import subprocess
import statistics as st

IIO = "/sys/bus/iio/devices/iio:device0"
PSY = "/sys/class/power_supply"


def channels():
    ch = {}
    for f in sorted(glob.glob(os.path.join(IIO, "in_current_*_input"))):
        ch[os.path.basename(f).replace("in_current_", "").replace("_input", "")] = f
    for f in sorted(glob.glob(os.path.join(IIO, "in_voltage_*_input"))):
        ch["V_" + os.path.basename(f).replace("in_voltage_", "").replace("_input", "")] = f
    for d in glob.glob(os.path.join(PSY, "*")):
        for k, n in (("current_now", "I"), ("voltage_now", "V")):
            p = os.path.join(d, k)
            if os.path.exists(p):
                ch[f"psy_{os.path.basename(d)}_{n}"] = p
    return ch


def read_all(ch):
    out = {}
    for k, p in ch.items():
        try:
            out[k] = int(open(p).read().strip())
        except Exception:
            out[k] = None
    return out


def npu_temp():
    v = []
    for d in glob.glob("/sys/class/thermal/thermal_zone*"):
        try:
            if "nspss" in open(os.path.join(d, "type")).read():
                v.append(int(open(os.path.join(d, "temp")).read()) / 1000.0)
        except Exception:
            pass
    return round(max(v), 1) if v else None


# --------------------------- charges ---------------------------------------
class CpuLoad:
    def __init__(self, n=None):
        self.n = n or os.cpu_count()
        self.p = []

    def start(self):
        for _ in range(self.n):
            self.p.append(subprocess.Popen(
                ["bash", "-c", "while :; do :; done"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))

    def stop(self):
        for p in self.p:
            p.terminate()
        for p in self.p:
            p.wait()
        self.p = []


class NpuLoad:
    """Charge NPU réaliste : invokes dos-à-dos dans un sous-processus."""

    SRC = """
import os, sys, numpy as np
QAIRT = os.environ.get("QAIRT", "/opt/qcom/qairt-new/qairt/2.48.0.260626")
os.environ["ADSP_LIBRARY_PATH"] = (f"{QAIRT}/lib/hexagon-v73/unsigned;"
  "/opt/qcom/qirp-sdk/lib/aarch64-oe-linux-gcc11.2;"
  "/opt/qcom/qirp-sdk/lib/hexagon-v73/unsigned;" + os.environ.get("ADSP_LIBRARY_PATH",""))
import ai_edge_litert.interpreter as tflite
d = tflite.load_delegate(f"{QAIRT}/lib/aarch64-oe-linux-gcc11.2/libQnnTFLiteDelegate.so",
                         options={"backend_type":"htp","htp_performance_mode":"1"})
i = tflite.Interpreter(model_path=sys.argv[1], experimental_delegates=[d])
i.allocate_tensors()
inp = i.get_input_details()[0]
dt, sh = inp["dtype"], inp["shape"]
t = (np.random.randint(0,255,sh).astype(dt) if dt.__name__ in ("uint8","int8")
     else np.random.randn(*sh).astype(dt))
while True:
    i.set_tensor(inp["index"], t); i.invoke()
"""

    def __init__(self, model):
        self.model = model
        self.p = None

    def start(self):
        open("/tmp/_npu_load.py", "w").write(self.SRC)
        self.p = subprocess.Popen(["python3", "/tmp/_npu_load.py", self.model],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(8)  # laisser le temps du chargement + allocate_tensors

    def stop(self):
        if self.p:
            self.p.terminate()
            self.p.wait()


# --------------------------- protocole -------------------------------------
def phase(ch, label, dur, rate, writer, rows):
    n = int(dur * rate)
    for _ in range(n):
        t = time.time()
        r = read_all(ch)
        row = {"t": round(t, 2), "phase": label, "npu_C": npu_temp()}
        row.update(r)
        writer.writerow(row)
        rows.append(row)
        time.sleep(max(0, 1.0 / rate - (time.time() - t)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", choices=["cpu", "npu", "none"], default="cpu")
    ap.add_argument("--model", default=None, help="requis si --load npu")
    ap.add_argument("--dur", type=float, default=60, help="durée de chaque phase (s)")
    ap.add_argument("--rate", type=float, default=2.0)
    a = ap.parse_args()

    ch = channels()
    if not ch:
        print("Aucun canal de courant trouvé.")
        return
    print(f"{len(ch)} canaux : {', '.join(list(ch)[:8])}...")

    gov = open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read().strip()
    print(f"gouverneur CPU = {gov}"
          + ("  <-- FIGE-LE EN 'performance' AVANT DE MESURER" if gov != "performance" else ""))

    load = {"cpu": CpuLoad, "npu": lambda: NpuLoad(a.model), "none": None}[a.load]
    load = load() if load else None
    if a.load == "npu" and not a.model:
        print("--load npu exige --model"); return

    out = f"power_probe_{a.load}.csv"
    rows = []
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t", "phase", "npu_C"] + list(ch))
        w.writeheader()
        print(f"[1/3] idle {a.dur:.0f}s..."); phase(ch, "idle_pre", a.dur, a.rate, w, rows)
        if load:
            print(f"[2/3] charge {a.load} {a.dur:.0f}s..."); load.start()
        phase(ch, "load", a.dur, a.rate, w, rows)
        if load:
            load.stop()
        print(f"[3/3] idle {a.dur:.0f}s..."); phase(ch, "idle_post", a.dur, a.rate, w, rows)

    # ------------------- verdict -------------------
    def vals(p, k):
        return [r[k] for r in rows if r["phase"] == p and r[k] is not None]

    print(f"\n{'canal':<38} {'idle (uA)':>14} {'load (uA)':>14} {'delta':>10} {'sigma':>9} verdict")
    print("-" * 100)
    usable = []
    for k in ch:
        a_i, a_l = vals("idle_pre", k) + vals("idle_post", k), vals("load", k)
        if len(a_i) < 5 or len(a_l) < 5:
            continue
        mi, ml = st.mean(a_i), st.mean(a_l)
        sd = st.pstdev(a_i) or 1e-9
        d = ml - mi
        ok = abs(d) > 3 * sd and abs(d) > 5000  # 3 sigma ET > 5 mA
        if ok:
            usable.append((k, d, sd))
        print(f"{k:<38} {mi:>14.0f} {ml:>14.0f} {d:>10.0f} {sd:>9.0f} "
              f"{'EXPLOITABLE' if ok else '-'}")

    # somme des entrées : souvent le seul agrégat qui a un sens physique
    iin = [k for k in ch if k.endswith("iin_smb") or k.endswith("iin_fb")]
    if iin:
        s_i = [sum(r[k] for k in iin if r[k] is not None)
               for r in rows if r["phase"].startswith("idle")]
        s_l = [sum(r[k] for k in iin if r[k] is not None)
               for r in rows if r["phase"] == "load"]
        if s_i and s_l:
            sd = st.pstdev(s_i) or 1e-9
            d = st.mean(s_l) - st.mean(s_i)
            print(f"\nSOMME des entrees ({'+'.join(iin)}) :")
            print(f"  idle={st.mean(s_i):.0f} uA  load={st.mean(s_l):.0f} uA  "
                  f"delta={d:.0f} uA  sigma={sd:.0f}  ->  "
                  f"{'SIGNAL REEL' if abs(d) > 3*sd else 'INDISTINGUABLE DU BRUIT'}")

    print(f"\n-> {out}")
    if usable:
        print("Au moins un canal repond : energie par inference mesurable en interne.")
    else:
        print("Aucun canal ne repond de facon significative.")
        print("=> Conclusion pour l'article : pas d'instrumentation de puissance")
        print("   exposee sur cette carte ; utiliser un wattmetre USB-PD inline")
        print("   (~20 EUR) ou renoncer aux metriques energetiques.")


if __name__ == "__main__":
    main()