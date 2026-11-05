"""
AI Analysis subsystem for the AUTOMATA orchestrator (MQTT).

Implements the `ai_analysis` subsystem contract of
https://github.com/GDqbrobotics/Automata_Project_Orchestrator
(backend/config/subsystems/ai_analysis.yaml + mqtt_formats.yaml):

    subscribes  automata/subsystem/ai_analysis/event     commands
    publishes   automata/subsystem/ai_analysis/status    state + heartbeat
    publishes   automata/subsystem/ai_analysis/response  points of interest

State machine: initializing -> ready -> (plan) analyzing -> ready
               analyzing -(stop)-> stopped -(plan|ready)-> ...
               any -(error)-> error -(reset)-> ready

On `plan` (carrying object_id) it runs the three pipeline steps on
<data_root>/<object_id>/ and writes their outputs to
<data_root>/<object_id>/HSI/processed_data/{roi,roi_clustered,crosshair}/:

    1. hsi/segment_roi_birefnet.py        automatic artifact mask
    2. hsi/run_roi_clustering.py          spectral clusters
    3. photogrammetry/analyze_hsi_clusters.py   XRF/Raman targets

then publishes all targets as 6D poses [x, y, z, roll, pitch, yaw]:
metres in the mesh frame (robot base_link for the 3DI meshes), radians,
roll/pitch/yaw = fixed-axis XYZ (R = Rz(yaw) Ry(pitch) Rx(roll)). The pose
frame's Z axis points INTO the surface (approach direction = -normal); its
X axis is the mesh-frame X axis projected on the surface plane (Y if X is
nearly parallel to the normal). Points are labelled `xrf` / `raman`, in
measurement order (XRF by rank; Raman by cluster, then rank). The points are
published BEFORE the `ready` status, so they are stored when the
orchestrator commands the measurement.

Usage:
    python ai_analysis_service.py --config ai_analysis_config.json
    python ai_analysis_service.py --data-root D:/data/batch1 --broker 192.168.1.10
"""

import argparse
import json
import logging
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

try:
    import paho.mqtt.client as mqtt
except ImportError:
    sys.exit("paho-mqtt is required: pip install paho-mqtt")

REPO = Path(__file__).resolve().parent
log = logging.getLogger("ai_analysis")

DEFAULTS = {
    "name": "ai_analysis",
    "broker": "localhost",
    "port": 1883,
    "topic_prefix": "automata/subsystem",
    "heartbeat_interval": 2.0,
    "data_root": "dataset",
    "python": sys.executable,
    "overwrite": False,
    "step_timeout_s": 1800,
    "segment_args": ["--hsi_only"],
    "cluster_args": ["--source", "raw"],
    "points_args": [],
    "fields": {
        "command_event": "event",
        "command_object_id": "object_id",
        "status_state": "state",
        "status_ready": "ready",
        "status_paused": "paused",
        "status_name": "name",
        "response_type": "type",
        "response_object_id": "object_id",
        "response_points": "points",
    },
}


def load_config(args):
    cfg = json.loads(json.dumps(DEFAULTS))
    if args.config:
        user = json.loads(Path(args.config).read_text(encoding="utf-8"))
        fields = user.pop("fields", {})
        cfg.update(user)
        cfg["fields"].update(fields)
    for key in ("broker", "port", "data_root", "python"):
        val = getattr(args, key.replace("-", "_"), None)
        if val is not None:
            cfg[key] = val
    if args.overwrite:
        cfg["overwrite"] = True
    cfg["data_root"] = str(Path(cfg["data_root"]).expanduser().resolve())
    return cfg


# ── Pose conversion ────────────────────────────────────────────────────────────

def pose_from_point_normal(point_cm, normal):
    """[x, y, z, roll, pitch, yaw]: metres, radians, Z axis = -normal."""
    n = np.asarray(normal, dtype=float)
    n /= np.linalg.norm(n) + 1e-20
    z = -n
    ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = ref - z * float(ref @ z)
    x /= np.linalg.norm(x) + 1e-20
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    pitch = math.asin(-max(-1.0, min(1.0, R[2, 0])))
    if abs(math.cos(pitch)) > 1e-6:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    p = np.asarray(point_cm, dtype=float) / 100.0
    return [round(float(v), 6) for v in (*p, roll, pitch, yaw)]


def rpy_to_matrix(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch),
                              math.sin(pitch), math.cos(yaw), math.sin(yaw))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def build_points(crosshair_dir, object_id, folder_name):
    """Points list for the response, from the step-3 JSON reports."""
    xrf_path = crosshair_dir / f"{folder_name}_xrf_report.json"
    raman_path = crosshair_dir / f"{folder_name}_raman_targets.json"
    points = []
    if xrf_path.exists():
        xrf = json.loads(xrf_path.read_text())
        for rank, t in enumerate(xrf.get("targets", []), 1):
            points.append({"label": "xrf",
                           "coordinates": pose_from_point_normal(t["point"], t["normal"]),
                           "rank": rank,
                           "normal": t["normal"]})
    if raman_path.exists():
        raman = json.loads(raman_path.read_text())
        for res in raman.get("results", []):
            if res.get("skipped"):
                continue
            for rank, t in enumerate(res.get("targets", []), 1):
                points.append({"label": "raman",
                               "coordinates": pose_from_point_normal(t["point"], t["normal"]),
                               "rank": rank,
                               "cluster": res.get("cluster"),
                               "shared_with_xrf": bool(t.get("from_xrf")),
                               "normal": t["normal"]})
    return points


# ── Service ────────────────────────────────────────────────────────────────────

class AIAnalysisService:
    def __init__(self, cfg):
        self.cfg = cfg
        self.f = cfg["fields"]
        prefix = f'{cfg["topic_prefix"]}/{cfg["name"]}'
        self.t_event, self.t_status, self.t_response = (
            f"{prefix}/event", f"{prefix}/status", f"{prefix}/response")
        self.state = "initializing"
        self.paused = False
        self.lock = threading.Lock()
        self.proc = None
        self.job = None
        self.cancelled = False
        self.running = True
        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                                      client_id=f'{cfg["name"]}-{os.getpid()}')
        except AttributeError:
            self.client = mqtt.Client(client_id=f'{cfg["name"]}-{os.getpid()}')
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.will_set(self.t_status, json.dumps(
            {self.f["status_state"]: "error", self.f["status_ready"]: False,
             self.f["status_paused"]: False, self.f["status_name"]: cfg["name"]}), qos=1)

    # MQTT plumbing
    def start(self):
        self.client.connect(self.cfg["broker"], int(self.cfg["port"]))
        self.client.loop_start()
        threading.Thread(target=self._heartbeat, daemon=True).start()
        self._initialize()

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        client.subscribe(self.t_event, qos=1)
        log.info("connected to %s:%s, listening on %s",
                 self.cfg["broker"], self.cfg["port"], self.t_event)
        self.publish_status()

    def publish_status(self):
        self.client.publish(self.t_status, json.dumps({
            self.f["status_state"]: self.state,
            self.f["status_ready"]: self.state == "ready",
            self.f["status_paused"]: self.paused,
            self.f["status_name"]: self.cfg["name"],
        }), qos=1)

    def _heartbeat(self):
        while self.running:
            time.sleep(float(self.cfg["heartbeat_interval"]))
            self.publish_status()

    def _set_state(self, state):
        with self.lock:
            prev, self.state = self.state, state
        log.info("%s -> %s", prev, state)
        if state not in ("analyzing",):
            self.paused = False
        self.publish_status()

    def _initialize(self):
        missing = [s for s in ("hsi/segment_roi_birefnet.py", "hsi/run_roi_clustering.py",
                               "photogrammetry/analyze_hsi_clusters.py")
                   if not (REPO / s).is_file()]
        root = Path(self.cfg["data_root"])
        if missing or not root.is_dir():
            log.error("init failed: missing %s / data_root %s", missing, root)
            self._set_state("error")
            return
        log.info("data root: %s", root)
        self._set_state("ready")

    # Commands
    def _on_message(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode())
        except (ValueError, UnicodeDecodeError):
            log.warning("ignoring non-JSON command")
            return
        event = data.get(self.f["command_event"])
        object_id = data.get(self.f["command_object_id"])
        log.info("command '%s'%s", event, f" (object {object_id})" if object_id else "")
        if event == "pause":
            self.paused = True
            self.publish_status()
        elif event == "resume":
            self.paused = False
            self.publish_status()
        elif event == "plan":
            if self.state not in ("ready", "stopped"):
                log.warning("plan rejected in state %s", self.state)
                return
            if not object_id:
                log.error("plan without object_id")
                self._set_state("error")
                return
            self.cancelled = False
            self._set_state("analyzing")
            self.job = threading.Thread(target=self._run_job, args=(object_id,), daemon=True)
            self.job.start()
        elif event == "stop":
            if self.state == "analyzing":
                self.cancelled = True
                self._kill()
                self._set_state("stopped")
        elif event == "ready" and self.state == "stopped":
            self._set_state("ready")
        elif event == "reset" and self.state == "error":
            self._set_state("ready")
        elif event == "error":
            self.cancelled = True
            self._kill()
            self._set_state("error")
        else:
            log.info("command '%s' ignored in state %s", event, self.state)

    def _kill(self):
        p = self.proc
        if p and p.poll() is None:
            p.terminate()
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()

    # Pipeline
    def _run_step(self, label, script, args):
        cmd = [self.cfg["python"], str(REPO / script), *args]
        log.info("[%s] %s", label, " ".join(cmd))
        t0 = time.time()
        self.proc = subprocess.Popen(cmd, cwd=REPO)
        try:
            rc = self.proc.wait(timeout=float(self.cfg["step_timeout_s"]))
        except subprocess.TimeoutExpired:
            self._kill()
            raise RuntimeError(f"{label} timed out")
        if self.cancelled:
            raise InterruptedError
        if rc != 0:
            raise RuntimeError(f"{label} failed (exit {rc})")
        log.info("[%s] done in %.1fs", label, time.time() - t0)

    def _run_job(self, object_id):
        root = Path(self.cfg["data_root"])
        obj = root / object_id
        out = obj / "HSI" / "processed_data"
        roi, clu, xh = out / "roi", out / "roi_clustered", out / "crosshair"
        ow = bool(self.cfg["overwrite"])
        t0 = time.time()
        try:
            if not (obj / "HSI").is_dir():
                raise RuntimeError(f"no HSI data for {object_id} under {obj}")
            mask = roi / f"{object_id}_mask.png"
            if ow or not mask.is_file():
                self._run_step("segment", "hsi/segment_roi_birefnet.py",
                               ["--batch", "--dataset", str(root), "--artifacts", object_id,
                                "--output_dir", str(roi), "--overwrite", *self.cfg["segment_args"]])
            if not mask.is_file():
                raise RuntimeError("segmentation produced no mask")
            clusters = clu / f"{object_id}_clusters.png"
            if ow or not clusters.is_file():
                self._run_step("cluster", "hsi/run_roi_clustering.py",
                               ["--dataset", str(root), "--roi", str(roi), "--output", str(clu),
                                "--artifacts", object_id, *self.cfg["cluster_args"]])
            if not clusters.is_file():
                raise RuntimeError("clustering produced no cluster map")
            xrf_report = xh / f"{object_id}_xrf_report.json"
            if ow or not xrf_report.is_file():
                self._run_step("points", "photogrammetry/analyze_hsi_clusters.py",
                               ["--artifact", object_id, "--dataset", str(root),
                                "--roi-dir", str(roi), "--clusters-dir", str(clu),
                                "--output", str(xh), *self.cfg["points_args"]])
            points = build_points(xh, object_id, object_id)
            if not points:
                raise RuntimeError("no measurement points produced")
            (xh / f"{object_id}_ai_points.json").write_text(json.dumps(
                {"object_id": object_id, "units": "m, rad",
                 "convention": "fixed-axis XYZ roll/pitch/yaw, pose Z = -surface normal",
                 "points": points}, indent=2))
        except InterruptedError:
            log.info("job for %s cancelled", object_id)
            return
        except Exception as e:
            log.error("job for %s failed: %s", object_id, e)
            if not self.cancelled:
                self._set_state("error")
            return
        if self.cancelled:
            return
        n_x = sum(p["label"] == "xrf" for p in points)
        self.client.publish(self.t_response, json.dumps({
            self.f["response_type"]: "object_points",
            self.f["response_object_id"]: object_id,
            self.f["response_points"]: points,
        }), qos=1).wait_for_publish(10)
        log.info("published %d points for %s (%d xrf, %d raman) in %.1fs",
                 len(points), object_id, n_x, len(points) - n_x, time.time() - t0)
        self._set_state("ready")

    def stop(self):
        self.running = False
        self.cancelled = True
        self._kill()
        self.client.loop_stop()
        self.client.disconnect()


def main():
    ap = argparse.ArgumentParser(description="AUTOMATA AI Analysis MQTT subsystem")
    ap.add_argument("--config", help="JSON config (see ai_analysis_config.example.json)")
    ap.add_argument("--broker")
    ap.add_argument("--port", type=int)
    ap.add_argument("--data-root", dest="data_root",
                    help="Folder containing one sub-folder per object id")
    ap.add_argument("--python", help="Python interpreter for the pipeline steps")
    ap.add_argument("--overwrite", action="store_true",
                    help="Recompute every step even if its output exists")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    svc = AIAnalysisService(load_config(args))
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    svc.start()
    while not stop.is_set():
        time.sleep(0.2)
    svc.stop()


if __name__ == "__main__":
    main()
