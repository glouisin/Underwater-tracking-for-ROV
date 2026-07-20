# -----------------------------------------------------------------------------
# monitoring.py — Instrumentation layer for the ROV vision pipeline
#
# Goal: produce, in a single run, every number needed to defend a systems /
# application paper:
#   1. Per-stage latency breakdown (capture, preprocess, NPU inference per model,
#      postprocess, NMS, tracking, rendering, encode/push)  -> latency CSV
#   2. Throughput: true camera FPS, loop rate, dropped/duplicated frames
#   3. Platform resources: per-core CPU%, process CPU%, RSS, CPU/GPU/NPU
#      frequencies, thermal zones (incl. nspss-0..3), GPU busy%, power rails
#   4. Control-quality signals: dx, dy, ez, z_raw, z_f, bbox area, confidence
#   5. A reproducibility snapshot (models, quantization params, delegate options,
#      kernel, SoC, git commit, all tunable constants)
#
# Design constraints:
#   - Zero third-party dependency (no psutil): everything read from /proc, /sys.
#   - Never block the control loop: CSV rows go through a queue to a writer
#     thread; system counters are sampled by a separate low-rate thread.
#   - Overhead measured and logged itself (t_monitor_us column).
#
# Usage (see integration_patch.md):
#     from monitoring import RunMonitor
#     mon = RunMonitor(run_tag="htp_dav3", config=CONFIG_DICT)
#     ...
#     with mon.stage("yolo_invoke"):
#         yolo_interpreter.invoke()
#     ...
#     mon.frame_row(frame_cnt=frame_cnt, seq=seq, ...)
#     mon.close()
# -----------------------------------------------------------------------------

import os
import csv
import json
import glob
import time
import queue
import socket
import threading
import subprocess
from collections import OrderedDict

__all__ = ["RunMonitor"]

_CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
_PAGE_KB = 4


# =============================================================================
#  Low-level /proc and /sys readers
# =============================================================================
def _read(path, default=None):
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except Exception:
        return default


def _read_int(path, default=None):
    v = _read(path)
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


class _CpuStat:
    """Per-core utilisation from /proc/stat deltas."""

    def __init__(self):
        self.prev = self._snapshot()

    @staticmethod
    def _snapshot():
        out = {}
        txt = _read("/proc/stat", "")
        for line in txt.splitlines():
            if not line.startswith("cpu"):
                break
            parts = line.split()
            name = parts[0]
            vals = [int(x) for x in parts[1:]]
            idle = vals[3] + (vals[4] if len(vals) > 4 else 0)  # idle + iowait
            total = sum(vals)
            out[name] = (idle, total)
        return out

    def sample(self):
        """Returns {'cpu': pct, 'cpu0': pct, ...} since previous call."""
        cur = self._snapshot()
        res = {}
        for name, (idle, total) in cur.items():
            pidle, ptotal = self.prev.get(name, (idle, total))
            d_total = total - ptotal
            d_idle = idle - pidle
            res[name] = 100.0 * (1.0 - d_idle / d_total) if d_total > 0 else 0.0
        self.prev = cur
        return res


class _ProcStat:
    """CPU time and memory of the current process (all threads)."""

    def __init__(self):
        self.prev_cpu = self._cpu_seconds()
        self.prev_t = time.time()

    @staticmethod
    def _cpu_seconds():
        txt = _read("/proc/self/stat", "")
        if not txt:
            return 0.0
        # field 14 = utime, 15 = stime (1-indexed after the comm field)
        after = txt[txt.rfind(")") + 2:].split()
        try:
            return (int(after[11]) + int(after[12])) / float(_CLK_TCK)
        except (IndexError, ValueError):
            return 0.0

    @staticmethod
    def rss_mb():
        txt = _read("/proc/self/status", "")
        for line in txt.splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
        return None

    @staticmethod
    def n_threads():
        txt = _read("/proc/self/status", "")
        for line in txt.splitlines():
            if line.startswith("Threads:"):
                return int(line.split()[1])
        return None

    def sample(self):
        now = time.time()
        cpu = self._cpu_seconds()
        dt = now - self.prev_t
        pct = 100.0 * (cpu - self.prev_cpu) / dt if dt > 0 else 0.0
        self.prev_cpu, self.prev_t = cpu, now
        return {
            "proc_cpu_pct": pct,          # can exceed 100% (multi-threaded)
            "proc_rss_mb": self.rss_mb(),
            "proc_threads": self.n_threads(),
        }


class _Thermal:
    """
    All thermal zones, discovered by name.

    On the QCS8550 the NPU/Hexagon zones are named nspss-0 .. nspss-3
    (NOT 'npu' / 'hexagon' / 'cdsp' as most documentation suggests).
    We log every zone and let the analysis script pick; this also documents
    the naming convention for the paper.
    """

    def __init__(self):
        self.zones = []
        for d in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
            t = _read(os.path.join(d, "type"))
            if t:
                self.zones.append((t, os.path.join(d, "temp")))

    def sample(self):
        out = {}
        for name, path in self.zones:
            v = _read_int(path)
            if v is None:
                continue
            c = v / 1000.0
            # -40 / 0 / flat ~25 are sentinel values of inactive sensors:
            # kept in the CSV but flagged so the analysis can drop them.
            out["temp_" + name.replace("-", "_")] = round(c, 2)
        return out


class _Freq:
    """CPU cluster frequencies + devfreq (GPU, DDR, NPU when exposed)."""

    def __init__(self):
        self.cpus = sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq"))
        self.devfreq = []
        for d in sorted(glob.glob("/sys/class/devfreq/*")):
            self.devfreq.append((os.path.basename(d), os.path.join(d, "cur_freq")))

    def sample(self):
        out = {}
        for p in self.cpus:
            core = p.split("/cpu")[1].split("/")[0]
            v = _read_int(p)
            if v is not None:
                out[f"freq_cpu{core}_mhz"] = round(v / 1000.0, 1)
        for name, p in self.devfreq:
            v = _read_int(p)
            if v is not None:
                key = "freq_" + "".join(ch if ch.isalnum() else "_" for ch in name) + "_mhz"
                out[key] = round(v / 1e6, 1)
        return out


class _Gpu:
    """Adreno busy ratio via kgsl (two counters: busy, total, reset on read)."""

    BUSY = "/sys/class/kgsl/kgsl-3d0/gpubusy"
    PCT = "/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage"
    CLK = "/sys/class/kgsl/kgsl-3d0/gpuclk"

    def sample(self):
        out = {}
        txt = _read(self.BUSY)
        if txt:
            try:
                busy, total = (int(x) for x in txt.split()[:2])
                out["gpu_busy_pct"] = round(100.0 * busy / total, 2) if total else 0.0
            except (ValueError, IndexError):
                pass
        if "gpu_busy_pct" not in out:
            txt = _read(self.PCT)
            if txt:
                try:
                    out["gpu_busy_pct"] = float(txt.split()[0].rstrip("%"))
                except ValueError:
                    pass
        clk = _read_int(self.CLK)
        if clk:
            out["gpu_clk_mhz"] = round(clk / 1e6, 1)
        return out


class _Power:
    """
    Best-effort board power. Most dev boards expose nothing; when a rail is
    present we log V/I so the paper can quote J/inference instead of only ms.
    """

    def __init__(self):
        self.rails = []
        for d in glob.glob("/sys/class/power_supply/*"):
            if os.path.exists(os.path.join(d, "current_now")):
                self.rails.append((os.path.basename(d), d))
        # PMIC ADC channels (QCS8550: pm8550b + smb139x charge pumps). These are
        # NOT under /sys/class/power_supply and are the only candidates for an
        # on-board energy measurement on this platform.
        self.iio = []
        for f in sorted(glob.glob("/sys/bus/iio/devices/iio:device*/in_current_*_input")):
            name = os.path.basename(f).replace("in_current_", "").replace("_input", "")
            self.iio.append((name, f))

    def sample(self):
        out = {}
        for name, d in self.rails:
            i = _read_int(os.path.join(d, "current_now"))
            v = _read_int(os.path.join(d, "voltage_now"))
            if i is not None:
                out[f"pw_{name}_ma"] = round(i / 1000.0, 1)
            if v is not None:
                out[f"pw_{name}_mv"] = round(v / 1000.0, 1)
            if i is not None and v is not None:
                out[f"pw_{name}_mw"] = round(abs(i * v) / 1e9 * 1000, 1)
        tot_in = 0
        for name, f in self.iio:
            v = _read_int(f)
            if v is None:
                continue
            out[f"iio_{name}_ua"] = v
            if name.endswith("iin_smb") or name.endswith("iin_fb"):
                tot_in += v
        if tot_in:
            out["iio_iin_total_ua"] = tot_in
        return out


class _NpuProbe:
    """
    There is no public utilisation counter for the Hexagon NSP on this SoC.
    We probe the usual candidate paths once, record which exist, and otherwise
    fall back on the derived duty cycle:
        npu_duty = (t_yolo_invoke + t_depth_invoke) / wall_time
    which is the honest, defensible metric to report in the paper.
    """

    CANDIDATES = [
        "/sys/kernel/debug/fastrpc",
        "/sys/class/misc/fastrpc-cdsp",
        "/sys/kernel/debug/rpmsg",
        "/sys/devices/platform/soc/soc:qcom,cdsp-pil",
        "/sys/class/remoteproc",
    ]

    @classmethod
    def probe(cls):
        return {p: os.path.exists(p) for p in cls.CANDIDATES}


# =============================================================================
#  Background system sampler
# =============================================================================
class _SystemSampler(threading.Thread):
    def __init__(self, out_queue, period=1.0):
        super().__init__(daemon=True)
        self.q = out_queue
        self.period = period
        self.running = True
        self.cpu = _CpuStat()
        self.proc = _ProcStat()
        self.thermal = _Thermal()
        self.freq = _Freq()
        self.gpu = _Gpu()
        self.power = _Power()
        self.latest = {}
        self._t0 = time.time()

    def run(self):
        while self.running:
            t = time.time()
            row = OrderedDict()
            row["t_wall"] = round(t, 3)
            row["t_rel"] = round(t - self._t0, 3)
            cores = self.cpu.sample()
            row["cpu_total_pct"] = round(cores.pop("cpu", 0.0), 2)
            for k in sorted(cores, key=lambda s: int(s[3:])):
                row[f"cpu{k[3:]}_pct"] = round(cores[k], 2)
            row.update({k: (round(v, 2) if isinstance(v, float) else v)
                        for k, v in self.proc.sample().items()})
            row.update(self.freq.sample())
            row.update(self.gpu.sample())
            row.update(self.thermal.sample())
            row.update(self.power.sample())
            row["load1"] = float((_read("/proc/loadavg", "0 0 0").split() or ["0"])[0])
            self.latest = row
            self.q.put(("system", row))
            # keep the cadence stable regardless of read cost
            time.sleep(max(0.0, self.period - (time.time() - t)))

    def stop(self):
        self.running = False


# =============================================================================
#  CSV writer thread (never blocks the control loop)
# =============================================================================
class _Writer(threading.Thread):
    def __init__(self, paths, flush_every=50):
        super().__init__(daemon=True)
        self.q = queue.Queue(maxsize=10000)
        self.paths = paths                    # {"frame": path, "system": path}
        self.files, self.writers, self.headers = {}, {}, {}
        self.flush_every = flush_every
        self._n = 0
        self.running = True
        self.dropped = 0

    def put(self, kind, row):
        try:
            self.q.put_nowait((kind, row))
        except queue.Full:
            self.dropped += 1

    def run(self):
        while self.running or not self.q.empty():
            try:
                kind, row = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            if kind not in self.files:
                f = open(self.paths[kind], "w", newline="")
                self.files[kind] = f
                self.headers[kind] = list(row.keys())
                w = csv.DictWriter(f, fieldnames=self.headers[kind], extrasaction="ignore")
                w.writeheader()
                self.writers[kind] = w
            else:
                # new keys appearing mid-run would silently vanish; record it
                missing = [k for k in row if k not in self.headers[kind]]
                if missing:
                    row = {k: v for k, v in row.items() if k in self.headers[kind]}
            self.writers[kind].writerow(row)
            self._n += 1
            if self._n % self.flush_every == 0:
                for f in self.files.values():
                    f.flush()
        for f in self.files.values():
            try:
                f.flush()
                f.close()
            except Exception:
                pass

    def stop(self):
        self.running = False


# =============================================================================
#  Stage timer
# =============================================================================
class _Stage:
    __slots__ = ("mon", "name", "t0")

    def __init__(self, mon, name):
        self.mon = mon
        self.name = name

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        dt = (time.perf_counter() - self.t0) * 1000.0
        self.mon._stages[self.name] = self.mon._stages.get(self.name, 0.0) + dt
        return False


# =============================================================================
#  Public API
# =============================================================================
class RunMonitor:
    STAGES = [
        "capture_wait", "preproc_yolo", "preproc_depth",
        "yolo_invoke", "depth_invoke",
        "postproc_yolo", "nms", "tracking", "depth_sample",
        "render", "udp_send", "gst_push", "throttle_sleep",
    ]

    def __init__(self, run_tag=None, log_dir=None, config=None,
                 system_period=1.0, extra_env=None):
        self.run_tag = run_tag or os.environ.get("RUN_TAG") or time.strftime("run_%Y%m%d_%H%M%S")
        self.log_dir = log_dir or os.environ.get("LOG_DIR", "/root/ai-rov/logs")
        self.dir = os.path.join(self.log_dir, self.run_tag)
        os.makedirs(self.dir, exist_ok=True)

        self.p_frame = os.path.join(self.dir, "frames.csv")
        self.p_system = os.path.join(self.dir, "system.csv")
        self.p_meta = os.path.join(self.dir, "meta.json")
        self.p_summary = os.path.join(self.dir, "summary.json")

        self.writer = _Writer({"frame": self.p_frame, "system": self.p_system})
        self.writer.start()
        self.sysq = queue.Queue()
        self.sampler = _SystemSampler(self.sysq, period=system_period)
        self.sampler.start()
        self._pump = threading.Thread(target=self._pump_system, daemon=True)
        self._pump.start()

        self._stages = {}
        self._series = {s: [] for s in self.STAGES}
        self._series["t_frame_total"] = []
        self._t0 = time.time()
        self._t_prev_frame = None
        self._last_seq = -1
        self._n_frames = 0
        self._n_loops = 0
        self._n_dup = 0          # loop iterations with no new camera frame
        self._n_skipped = 0      # camera frames produced but never processed
        self._t_npu_busy = 0.0   # yolo_invoke + depth_invoke accumulated
        self._ctrl = []          # (t, dx, dy, ez) for stationkeeping metrics
        self._overhead_us = []

        self.meta = self._collect_meta(config, extra_env)
        with open(self.p_meta, "w") as f:
            json.dump(self.meta, f, indent=2, default=str)
        print(f"[MON] run_tag={self.run_tag}  ->  {self.dir}")

    # ---------------- meta / reproducibility ----------------
    def _collect_meta(self, config, extra_env):
        def sh(cmd):
            try:
                return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL,
                                               timeout=5).decode().strip()
            except Exception:
                return None

        cpuinfo = _read("/proc/cpuinfo", "")
        model = _read("/sys/firmware/devicetree/base/model", "").replace("\x00", "")
        meta = {
            "run_tag": self.run_tag,
            "t_start_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "host": socket.gethostname(),
            "board_model": model,
            "kernel": sh("uname -a"),
            "python": sh("python3 --version"),
            "n_cpus": os.cpu_count(),
            "cpu_part_lines": [l for l in cpuinfo.splitlines() if "CPU part" in l],
            "git_commit": sh("git rev-parse --short HEAD"),
            "git_dirty": bool(sh("git status --porcelain")),
            "qairt_path": os.environ.get("ADSP_LIBRARY_PATH", "").split(";")[0],
            "thermal_zones": [z[0] for z in _Thermal().zones],
            "npu_sysfs_probe": _NpuProbe.probe(),
            "governor": _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
            "config": config or {},
        }
        if extra_env:
            meta["env"] = {k: os.environ.get(k) for k in extra_env}
        return meta

    def register_model(self, name, interpreter, model_path, delegate_options=None):
        """Store input/output shapes + quantization params for the paper's table."""
        try:
            i = interpreter.get_input_details()[0]
            o = interpreter.get_output_details()[0]
            info = {
                "path": model_path,
                "size_mb": round(os.path.getsize(model_path) / 1e6, 2)
                if os.path.exists(model_path) else None,
                "input_shape": [int(x) for x in i["shape"]],
                "input_dtype": str(i["dtype"]),
                "input_quant": [float(i["quantization"][0]), int(i["quantization"][1])],
                "output_shape": [int(x) for x in o["shape"]],
                "output_dtype": str(o["dtype"]),
                "output_quant": [float(o["quantization"][0]), int(o["quantization"][1])],
                "delegate_options": delegate_options,
                "n_tensors": len(interpreter.get_tensor_details()),
            }
        except Exception as e:  # never let instrumentation kill the run
            info = {"path": model_path, "error": repr(e)}
        self.meta.setdefault("models", {})[name] = info
        with open(self.p_meta, "w") as f:
            json.dump(self.meta, f, indent=2, default=str)

    # ---------------- per-frame API ----------------
    def stage(self, name):
        return _Stage(self, name)

    def mark_loop(self, new_frame):
        self._n_loops += 1
        if not new_frame:
            self._n_dup += 1

    def frame_row(self, frame_cnt, seq=None, **fields):
        """
        Call once per processed frame, at the very end of the loop body.
        Everything accumulated by stage() since the last call is flushed here.
        """
        t_mon0 = time.perf_counter()
        now = time.time()
        st = self._stages
        self._stages = {}

        dt_frame = (now - self._t_prev_frame) * 1000.0 if self._t_prev_frame else 0.0
        self._t_prev_frame = now
        self._n_frames += 1

        if seq is not None and self._last_seq >= 0:
            gap = seq - self._last_seq
            if gap > 1:
                self._n_skipped += gap - 1
        if seq is not None:
            self._last_seq = seq

        t_npu = st.get("yolo_invoke", 0.0) + st.get("depth_invoke", 0.0)
        self._t_npu_busy += t_npu / 1000.0

        row = OrderedDict()
        row["t_wall"] = round(now, 4)
        row["t_rel"] = round(now - self._t0, 4)
        row["frame_cnt"] = frame_cnt
        row["seq"] = seq if seq is not None else -1
        row["dt_frame_ms"] = round(dt_frame, 3)
        row["fps_inst"] = round(1000.0 / dt_frame, 2) if dt_frame > 0 else 0.0
        for s in self.STAGES:
            row["t_" + s + "_ms"] = round(st.get(s, 0.0), 3)
        row["t_npu_ms"] = round(t_npu, 3)
        row["t_cpu_ms"] = round(sum(st.values()) - t_npu, 3)
        row["t_accounted_ms"] = round(sum(st.values()), 3)
        row["npu_duty"] = round(t_npu / dt_frame, 4) if dt_frame > 0 else 0.0

        # counters
        row["n_loops"] = self._n_loops
        row["n_dup_loops"] = self._n_dup
        row["n_skipped_frames"] = self._n_skipped

        # user-supplied application fields (detections, control, depth, ...)
        for k, v in fields.items():
            row[k] = v

        # last known system snapshot, joined inline for convenience
        for k, v in (self.sampler.latest or {}).items():
            if k in ("t_wall", "t_rel"):
                continue
            row["sys_" + k] = v

        for s in self.STAGES:
            self._series[s].append(st.get(s, 0.0))
        self._series["t_frame_total"].append(dt_frame)
        if "dx" in fields and fields.get("vision_valid", True):
            self._ctrl.append((now - self._t0, fields.get("dx"), fields.get("dy"),
                               fields.get("ez")))

        ov = (time.perf_counter() - t_mon0) * 1e6
        self._overhead_us.append(ov)
        row["t_monitor_us"] = round(ov, 1)
        self.writer.put("frame", row)

    # ---------------- internals ----------------
    def _pump_system(self):
        while True:
            try:
                kind, row = self.sysq.get(timeout=0.5)
                self.writer.put(kind, row)
            except queue.Empty:
                if not self.sampler.running:
                    return

    # ---------------- summary ----------------
    @staticmethod
    def _stats(v):
        v = [x for x in v if x is not None]
        if not v:
            return None
        s = sorted(v)
        n = len(s)

        def pct(p):
            return s[min(n - 1, int(round(p / 100.0 * (n - 1))))]

        return {
            "n": n,
            "mean": round(sum(s) / n, 3),
            "std": round((sum((x - sum(s) / n) ** 2 for x in s) / n) ** 0.5, 3),
            "min": round(s[0], 3),
            "p50": round(pct(50), 3),
            "p90": round(pct(90), 3),
            "p95": round(pct(95), 3),
            "p99": round(pct(99), 3),
            "max": round(s[-1], 3),
        }

    def close(self):
        elapsed = time.time() - self._t0
        summary = {
            "run_tag": self.run_tag,
            "duration_s": round(elapsed, 2),
            "frames_processed": self._n_frames,
            "loops": self._n_loops,
            "dup_loops": self._n_dup,
            "camera_frames_skipped": self._n_skipped,
            "fps_true": round(self._n_frames / elapsed, 3) if elapsed else 0.0,
            "loop_rate_hz": round(self._n_loops / elapsed, 3) if elapsed else 0.0,
            "frame_drop_ratio": round(
                self._n_skipped / max(1, self._n_skipped + self._n_frames), 4),
            "npu_duty_cycle": round(self._t_npu_busy / elapsed, 4) if elapsed else 0.0,
            "monitor_overhead_us": self._stats(self._overhead_us),
            "writer_rows_dropped": self.writer.dropped,
            "latency_ms": {k: self._stats(v) for k, v in self._series.items()
                           if any(x > 0 for x in v)},
        }

        # stationkeeping quality (the actual thesis claim)
        if self._ctrl:
            for i, name in ((1, "dx"), (2, "dy"), (3, "ez")):
                vals = [c[i] for c in self._ctrl if c[i] is not None]
                if not vals:
                    continue
                n = len(vals)
                rms = (sum(x * x for x in vals) / n) ** 0.5
                iae = sum(abs(x) for x in vals) / n
                summary.setdefault("control", {})[name] = {
                    "rms": round(rms, 5),
                    "mae": round(iae, 5),
                    "mean": round(sum(vals) / n, 5),
                    "abs_p95": round(sorted(abs(x) for x in vals)[int(0.95 * (n - 1))], 5),
                    "n": n,
                }

        with open(self.p_summary, "w") as f:
            json.dump(summary, f, indent=2)

        self.sampler.stop()
        time.sleep(0.3)
        self.writer.stop()
        self.writer.join(timeout=3)

        print(f"\n[MON] {self.run_tag}: {self._n_frames} frames / {elapsed:.1f}s "
              f"-> FPS_true={summary['fps_true']:.2f}  "
              f"NPU duty={summary['npu_duty_cycle']*100:.1f}%  "
              f"drop={summary['frame_drop_ratio']*100:.1f}%")
        print(f"[MON] files: {self.p_frame}\n              {self.p_system}\n"
              f"              {self.p_summary}")
        return summary