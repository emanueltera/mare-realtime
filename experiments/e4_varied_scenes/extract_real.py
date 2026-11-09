"""Turn a recorded flight's MCAP into a set the camera-format networks can be run on.

    python extract_real.py --mcap <flight>.mcap --navigation <flight>_frame_navigation.csv \
        --orientation-topic <topic> --out out/real/<name> [--min-seconds 60]

Needs the `mcap` package and OpenCV (the dragonfly environment has both). The MCAP is the
recorder's format: /camera/image (10 Hz JPEG, 720×1280 portrait, the flight direction to the
image's right), /camera/frame_metadata, /mavlink/attitude_quaternion (the stabilised image's
attitude, FRD→NED with x = portrait right = flight, y = portrait down, z = optical axis),
/mavlink/global_position_int (GNSS lat/lon, and `relative_alt`, the recording's fused height
above the sea), /mavlink/highres_imu (the camera's raw gyro, body FRD axes, the same labels)
and the camera's per-frame orientation topic (--orientation-topic; its `iori` is the
stabilisation's rotation of the image against the body). The camera's own frame counter is
the field ending in `_frame_index` in that topic and in /camera/frame_metadata.

The over-water stretches are the navigation table's `scored` runs of at least --min-seconds.
Per stretch, in the order of its frames:
  frames      (N, 288, 512) uint8 luma: the portrait image turned 90° anticlockwise (flight up the
              frame, as in the camera-format renders) and area-shrunk by 2.5, as e3.shrink does
  R           (N, 3, 3) image attitude, landscape camera axes (x right, y down, z optical) → ENU
  rot_gyro    (N−1, 3) the image's rotation between frames from the camera's own gyro and its
              stabilisation record: IORI_t · ∏exp(ω dt) · IORI_t+1ᵀ, landscape axes, radians;
              no bias removed
  rot_att     (N−1, 3) the same from the attitude, R_tᵀ R_t+1 (the attitude's heading uses GNSS)
  height      (N,) the recording's fused height above the sea, m, derived from the video using
              GNSS speed (so it is not GNSS-free; its scale is uncertain by ±10–15 % in the
              recording's own notes)
  en          (N, 2) GNSS position, east and north in metres from the stretch's first frame
  t           (N,) UTC seconds; frame (N,) the image's index in the MCAP
Written as <out>/<stretch>/{frames.npy, real.npz} and <out>/index.json.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from mcap.reader import make_reader
from scipy.spatial.transform import Rotation, Slerp

SIZE = (512, 288)  # (width, height) after shrinking the 1280 × 720 landscape frame
# Portrait camera axes → landscape: the image turns 90° anticlockwise, so portrait right
# (the flight direction) becomes landscape up and portrait down becomes landscape right.
M = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], float)
NED_TO_ENU = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], float)
WGS_A, WGS_E2 = 6378137.0, 6.69437999014e-3


def utc(s: str) -> float:
    return datetime.fromisoformat(s).timestamp()


def stretches(nav: Path, min_s: float) -> list[tuple[int, int]]:
    rows = list(csv.DictReader(open(nav)))
    out, start = [], None
    for k, r in enumerate(rows + [{"scored": "0"}]):
        on = r["scored"] == "1"
        if on and start is None:
            start = k
        if not on and start is not None:
            a, b = int(rows[start]["frame_index"]), int(rows[k - 1]["frame_index"])
            if float(rows[k - 1]["t_relative_s"]) - float(rows[start]["t_relative_s"]) >= min_s:
                out.append((a, b))
            start = None
    return out


def camera_frame_index(d: dict) -> int | None:
    keys = [k for k in d if k.endswith("_frame_index")]
    return d[keys[0]] if len(keys) == 1 else None


def read(mcap: Path, wanted: set[int], orientation_topic: str):
    """Every stream needed, as arrays on UTC seconds; frames only for wanted image indices."""
    att_t, att_q, pos_t, pos, imu_t, imu_w, ori = [], [], [], [], [], [], {}
    img = {}
    meta = []
    with open(mcap, "rb") as f:
        topics = ["/camera/image", "/camera/frame_metadata", "/mavlink/attitude_quaternion",
                  "/mavlink/global_position_int", "/mavlink/highres_imu", orientation_topic]
        n_img = 0
        for _, ch, msg in make_reader(f).iter_messages(topics=topics, log_time_order=True):
            tp = ch.topic
            if tp == "/camera/image":
                if n_img in wanted:
                    d = json.loads(msg.data)
                    buf = np.frombuffer(base64.b64decode(d["data"]), np.uint8)
                    im = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
                    if im is None or im.shape != (1280, 720):
                        raise RuntimeError(f"image {n_img}: unreadable or not 720×1280 portrait")
                    land = np.rot90(im, 1)  # anticlockwise: portrait right → landscape top
                    img[n_img] = (d["timestamp"]["sec"] + d["timestamp"]["nsec"] * 1e-9,
                                  cv2.resize(np.ascontiguousarray(land), SIZE, interpolation=cv2.INTER_AREA))
                n_img += 1
                continue
            d = json.loads(msg.data)
            if tp == "/camera/frame_metadata":
                meta.append((d["timestamp"], camera_frame_index(d)))
            elif tp == "/mavlink/attitude_quaternion":
                att_t.append(utc(d["utc_time"]))
                att_q.append([d["q2"], d["q3"], d["q4"], d["q1"]])  # scipy order x, y, z, w
            elif tp == "/mavlink/global_position_int":
                pos_t.append(utc(d["utc_time"]))
                pos.append([d["lat"] * 1e-7, d["lon"] * 1e-7, d["relative_alt"] * 1e-3])
            elif tp == "/mavlink/highres_imu":
                imu_t.append(utc(d["utc_time"]))
                imu_w.append([d["xgyro"], d["ygyro"], d["zgyro"]])
            elif tp == orientation_topic:
                q = d["iori"]
                ori[camera_frame_index(d)] = [q["x"], q["y"], q["z"], q["w"]]
    return dict(att_t=np.array(att_t), att_q=np.array(att_q), pos_t=np.array(pos_t), pos=np.array(pos),
                imu_t=np.array(imu_t), imu_w=np.array(imu_w), ori=ori, img=img, meta=meta)


def iori_frd(q_xyzw) -> np.ndarray:
    """IORI as recorded (raw optical axes x right, y down, z lens) relabelled to the body FRD
    axes the attitude and the gyro use: (w, x, y, z) → (w, y, −x, z), as the recording documents."""
    x, y, z, w = q_xyzw
    return Rotation.from_quat([y, -x, z, w]).as_matrix()


def gyro_between(imu_t, imu_w, t0, t1) -> np.ndarray:
    """Body rotation from t0 to t1, R_body(t0)ᵀ R_body(t1), integrating the gyro (zero-order hold)."""
    k0, k1 = np.searchsorted(imu_t, [t0, t1])
    ts = np.concatenate([[t0], imu_t[k0:k1], [t1]])
    ws = imu_w[max(k0 - 1, 0):k1]
    R = np.eye(3)
    for w, dt in zip(ws, np.diff(ts)):
        R = R @ Rotation.from_rotvec(np.asarray(w) * dt).as_matrix()
    return R


def local_en(lat, lon, lat0, lon0):
    s = math.sin(math.radians(lat0))
    n = WGS_A / math.sqrt(1 - WGS_E2 * s * s)
    m = WGS_A * (1 - WGS_E2) / (1 - WGS_E2 * s * s) ** 1.5
    return np.stack([np.radians(lon - lon0) * n * math.cos(math.radians(lat0)), np.radians(lat - lat0) * m], -1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcap", required=True)
    ap.add_argument("--navigation", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--orientation-topic", required=True, help="the camera's per-frame orientation topic (with `iori`)")
    ap.add_argument("--min-seconds", type=float, default=60)
    ap.add_argument("--max-stretches", type=int, default=None, help="only the first N stretches (for a quick check)")
    a = ap.parse_args()
    segs = stretches(Path(a.navigation), a.min_seconds)[:a.max_stretches]
    wanted = {i for s, e in segs for i in range(s, e + 1)}
    print(f"{len(segs)} stretches, {len(wanted)} frames; reading {a.mcap}", flush=True)
    d = read(Path(a.mcap), wanted, a.orientation_topic)
    slerp = Slerp(d["att_t"], Rotation.from_quat(d["att_q"]))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for k, (s, e) in enumerate(segs):
        idx = np.arange(s, e + 1)
        t = np.array([d["img"][i][0] for i in idx])
        frames = np.stack([d["img"][i][1] for i in idx])
        R_ned_p = slerp(t).as_matrix()                                  # portrait image → NED
        R = NED_TO_ENU @ R_ned_p @ M.T                                   # landscape image → ENU
        gfi = [d["meta"][i][1] for i in idx]
        if any(g is None or g not in d["ori"] for g in gfi):
            raise RuntimeError(f"stretch {k}: frames without a stabilisation record")
        Ii = [iori_frd(d["ori"][g]) for g in gfi]
        rot_gyro, rot_att = [], []
        for j in range(len(idx) - 1):
            Rb = gyro_between(d["imu_t"], d["imu_w"], t[j], t[j + 1])
            # image(FRD) = body · IORIᵀ, so image_tᵀ image_t+1 = IORI_t · body_tᵀ body_t+1 · IORI_t+1ᵀ
            Rp = Ii[j] @ Rb @ Ii[j + 1].T
            rot_gyro.append(Rotation.from_matrix(M @ Rp @ M.T).as_rotvec())
            rot_att.append(Rotation.from_matrix(R[j].T @ R[j + 1]).as_rotvec())
        lat = np.interp(t, d["pos_t"], d["pos"][:, 0])
        lon = np.interp(t, d["pos_t"], d["pos"][:, 1])
        height = np.interp(t, d["pos_t"], d["pos"][:, 2])
        en = local_en(lat, lon, lat[0], lon[0])
        name = f"stretch-{k:02d}"
        sd = out / name
        sd.mkdir(exist_ok=True)
        np.save(sd / "frames.npy", frames)
        np.savez(sd / "real.npz", R=R, rot_gyro=np.array(rot_gyro), rot_att=np.array(rot_att), height=height, en=en, t=t, frame=idx,
                 lat0=lat[0], lon0=lon[0])
        dt = np.diff(t)
        dif = np.degrees(np.linalg.norm(np.array(rot_gyro) - np.array(rot_att), axis=1))
        nose = R[:, :, 1] * -1                                            # image up in ENU
        v = np.diff(en, axis=0)
        cos = (nose[:-1, :2] * v).sum(1) / (np.linalg.norm(nose[:-1, :2], axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
        info = dict(name=name, frames=[int(s), int(e)], n=len(idx), seconds=float(t[-1] - t[0]),
                    dt_ms=[float(dt.min() * 1e3), float(np.median(dt) * 1e3), float(dt.max() * 1e3)],
                    path_m=float(np.linalg.norm(v, axis=1).sum()), height_m=[float(height.min()), float(np.median(height)), float(height.max())],
                    gyro_vs_att_deg=[float(np.median(dif)), float(np.percentile(dif, 90))],
                    up_vs_track_cos_median=float(np.median(cos)))
        index.append(info)
        print(json.dumps(info), flush=True)
    (out / "index.json").write_text(json.dumps(dict(mcap=Path(a.mcap).name, stretches=index), indent=1) + "\n")


if __name__ == "__main__":
    main()
