"""E3 data: consecutive frame pairs, the sensors an aircraft without GNSS carries, and the
camera's relative pose between the two frames.

Conventions (as E2's geom.py): world ENU, z up; camera OpenCV axes (x right, y down, z
along the optical axis); R_wc (frames.csv R00..R22) maps camera vectors to ENU; the camera
sits at cam_e, cam_n, cam_u. The camera is hard-mounted (gimbal.stabilized 0), so the mount
— the nose direction in camera axes — is a fixed calibration per run. Frames are E3's 288 × 512
portrait or the 512 × 288 landscape of the recorded flights' camera; `Camera` holds which.

Labels for the pair (t, t+1), both in the camera axes of frame t:
  dp   = R_wc,tᵀ (p_t+1 − p_t)                        translation, metres
  rot  = log(R_wc,tᵀ R_wc,t+1)                         rotation vector, radians

Sensors, each with a per-run bias and per-frame white noise (NOISE; drawn afresh per sample
in training, from a per-run seed for evaluation):
  attitude (AHRS)  R̃_wc = Exp(δ) R_wc, δ = tilt (E, N) and yaw (U) errors; the network gets
                   gravity in camera axes at t and t+1, R̃ᵀ(0, 0, −1); yaw does not enter it
  gyro             rotation vector of the pair + bias·dt + white noise (camera axes)
  baro             camera height above mean sea level + bias + noise
  airspeed         |v − wind| × (1 + scale error) + noise, along the nose (no direction sensor)
GNSS, wind and heading are not inputs. The network predicts dp / h̃ (h̃ = baro height at t)
and a correction to the gyro's rotation; `Pair.sensor` is the 14-vector it sees. With
airspeed=False its airspeed entries (10–13) are zero, for a platform without that sensor.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import resource

import numpy as np

from .fx import Fx, apply_fx, draw_fx

# Each run's frames are a memory map, which holds a file descriptor: 3000 runs need more
# than the usual soft limit of 1024.
_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
if _soft < 65536:
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, _hard), _hard))

DT = 0.1


@dataclass(frozen=True)
class Camera:
    """The pinhole camera of a set's shrunk frames: size and focal length in pixels, square
    pixels, principal point at the centre. A network is built for one camera and is only
    run on sets with that camera."""

    h: int
    w: int
    f: float

    @property
    def cx(self) -> float:
        return self.w / 2

    @property
    def cy(self) -> float:
        return self.h / 2

    def as_dict(self) -> dict:
        return dict(h=self.h, w=self.w, f=self.f)

    def matches(self, other: "Camera", rtol: float = 1e-4) -> bool:
        return self.h == other.h and self.w == other.w and abs(self.f - other.f) <= rtol * self.f


# E3's camera (frames.csv): fx = fy = 618.04, cx = 360, cy = 640 on 720 × 1280 portrait,
# shrunk by 2.5. Checkpoints saved without a camera were trained on it.
E3_CAM = Camera(512, 288, 618.040816 * (288 / 720))  # bit-equal to the scaled frames.csv focal
H, W, FX, CX, CY = E3_CAM.h, E3_CAM.w, E3_CAM.f, E3_CAM.cx, E3_CAM.cy

D = math.pi / 180
NOISE = dict(
    tilt_bias=0.3 * D, tilt_white=0.1 * D,      # rad
    yaw_bias=2.0 * D, yaw_white=0.3 * D,        # rad (used only to place steps in the world)
    gyro_bias=0.05 * D, gyro_white=0.003 * D,   # rad/s per axis; rad per step
    baro_bias=2.0, baro_white=0.3,              # m
    air_scale=0.02, air_white=0.3,              # fraction; m/s
)
NO_NOISE = {k: 0.0 for k in NOISE}


def parse_noise(specs: list[str] | None) -> dict:
    """NOISE with overrides 'key=value' (a value ending in 'd' is in degrees, e.g. yaw_bias=0.3d).
    Scaling a σ keeps every other draw the same: simulate draws in a fixed order."""
    noise = dict(NOISE)
    for spec in specs or []:
        k, v = spec.split("=", 1)
        if k not in noise:
            raise ValueError(f"unknown noise key {k}; one of {sorted(noise)}")
        noise[k] = float(v[:-1]) * D if v.endswith("d") else float(v)
    return noise
SENSOR_DIM = 14
FX_STREAM = 0xf8  # keeps the camera effects' draws apart from the sensors'


# ---- rotations ------------------------------------------------------------------------------


def exp_so3(w: np.ndarray) -> np.ndarray:
    """Rotation matrices for rotation vectors w (..., 3)."""
    w = np.asarray(w, np.float64)
    th = np.linalg.norm(w, axis=-1, keepdims=True)[..., None]
    k = np.zeros(w.shape[:-1] + (3, 3))
    k[..., 0, 1], k[..., 0, 2], k[..., 1, 2] = -w[..., 2], w[..., 1], -w[..., 0]
    k[..., 1, 0], k[..., 2, 0], k[..., 2, 1] = w[..., 2], -w[..., 1], w[..., 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        a = np.where(th > 1e-12, np.sin(th) / th, 1.0)
        b = np.where(th > 1e-12, (1 - np.cos(th)) / th**2, 0.5)
    return np.eye(3) + a * k + b * (k @ k)


def log_so3(R: np.ndarray) -> np.ndarray:
    """Rotation vectors (..., 3) of rotation matrices (angles below π)."""
    R = np.asarray(R, np.float64)
    c = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    th = np.arccos(c)[..., None]
    v = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(th > 1e-9, th / (2 * np.sin(th)), 0.5)
    return f * v


# ---- runs -----------------------------------------------------------------------------------


def _read(path: Path) -> dict[str, np.ndarray]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def _npy_shape(path: Path) -> tuple[int, ...]:
    """The shape stored in a .npy file's header, without opening the array."""
    with open(path, "rb") as f:
        major, _ = np.lib.format.read_magic(f)
        read = np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0
        return read(f)[0]


@dataclass
class Run:
    """One rendered run: frames (memory-mapped) and its truth."""

    id: str
    dir: Path
    R: np.ndarray             # (N, 3, 3) R_wc
    p: np.ndarray             # (N, 3) camera position, ENU m
    airspeed: np.ndarray      # (N,) m/s, true
    nose: np.ndarray          # (3,) nose direction in camera axes (the mount)
    seed: int
    sea_rate: float
    meta: dict
    full: dict                # the full-size camera from frames.csv: fx, fy, cx, cy, width, height

    @property
    def cam(self) -> Camera:
        """The camera of frames.npy: frames.csv's intrinsics scaled to the shrunk size."""
        if "_cam" not in self.__dict__:
            n, h, w = _npy_shape(self.dir / "frames.npy")
            c = self.full
            s = w / c["width"]
            ok = (abs(h / c["height"] - s) < 1e-9 and abs(c["fx"] - c["fy"]) < 1e-6 * c["fx"]
                  and abs(c["cx"] - c["width"] / 2) < 1e-6 and abs(c["cy"] - c["height"] / 2) < 1e-6)
            if not ok or n != self.n:
                raise ValueError(f"{self.dir}: frames {n}×{h}×{w} do not fit frames.csv's camera {c} and {self.n} poses")
            self.__dict__["_cam"] = Camera(int(h), int(w), c["fx"] * s)
        return self.__dict__["_cam"]

    @property
    def frames(self) -> np.ndarray:
        """(N, H, W) uint8, memory-mapped on first use (per process)."""
        if "_frames" not in self.__dict__:
            self.__dict__["_frames"] = np.load(self.dir / "frames.npy", mmap_mode="r")
        return self.__dict__["_frames"]

    @property
    def n(self) -> int:
        return len(self.p)

    @classmethod
    def load(cls, d: Path) -> "Run":
        d = Path(d)
        fr = _read(d / "frames.csv")
        tr = _read(d / "trajectory.csv")
        run = json.loads((d / "run.json").read_text())
        R = np.stack([np.stack([fr[f"R{i}{j}"] for j in range(3)], -1) for i in range(3)], -2)
        p = np.stack([fr["cam_e"], fr["cam_n"], fr["cam_u"]], -1)
        psi, th = np.radians(tr["heading"][0]), np.radians(tr["pitch"][0])
        nose = R[0].T @ np.array([math.sin(psi) * math.cos(th), math.cos(psi) * math.cos(th), math.sin(th)])
        assert len(p) == len(tr["airspeed"]), d
        full = {k: float(fr[k][0]) for k in ("fx", "fy", "cx", "cy", "width", "height")}
        return cls(id=d.name, dir=d, R=R, p=p, airspeed=tr["airspeed"], nose=nose / np.linalg.norm(nose),
                   seed=int(run["trajectory"]["seed"]), sea_rate=float(run.get("render", {}).get("seaRate", 1)), meta=run, full=full)

    def labels(self) -> tuple[np.ndarray, np.ndarray]:
        """dp (N−1, 3) metres and rot (N−1, 3) radians, camera axes of the first frame."""
        dp = np.einsum("nji,nj->ni", self.R[:-1], self.p[1:] - self.p[:-1])
        rot = log_so3(np.einsum("nji,njk->nik", self.R[:-1], self.R[1:]))
        return dp, rot


@dataclass
class Sensors:
    """Simulated sensor readings for every frame of a run (one noise draw)."""

    R_ahrs: np.ndarray    # (N, 3, 3) AHRS attitude of the camera (includes yaw error)
    gyro: np.ndarray      # (N−1, 3) rotation vector per step, camera axes
    baro: np.ndarray      # (N,) m
    airspeed: np.ndarray  # (N,) m/s


def simulate(run: Run, rng: np.random.Generator, noise: dict = NOISE) -> Sensors:
    n = run.n
    tilt = rng.normal(0, noise["tilt_bias"], 2) + rng.normal(0, noise["tilt_white"], (n, 2))
    yaw = rng.normal(0, noise["yaw_bias"]) + rng.normal(0, noise["yaw_white"], n)
    R_ahrs = exp_so3(np.concatenate([tilt, yaw[:, None]], 1)) @ run.R
    _, rot = run.labels()
    gyro = rot + rng.normal(0, noise["gyro_bias"], 3) * DT + rng.normal(0, noise["gyro_white"], (n - 1, 3))
    baro = run.p[:, 2] + rng.normal(0, noise["baro_bias"]) + rng.normal(0, noise["baro_white"], n)
    air = run.airspeed * (1 + rng.normal(0, noise["air_scale"])) + rng.normal(0, noise["air_white"], n)
    return Sensors(R_ahrs=R_ahrs, gyro=gyro, baro=baro, airspeed=air)


def sensor_vector(run: Run, s: Sensors, i: int) -> np.ndarray:
    """What the network sees beside the two images for the pair (i, i+1)."""
    down = np.array([0.0, 0.0, -1.0])
    g0 = s.R_ahrs[i].T @ down
    g1 = s.R_ahrs[i + 1].T @ down
    h = max(s.baro[i], 5.0)
    air_step = run.nose * s.airspeed[i] * DT / h        # airspeed displacement in heights
    return np.concatenate([g0, g1, s.gyro[i] / (0.05), [math.log(h / 100)], air_step * 10, [s.airspeed[i] / 40]]).astype(np.float32)


def run_seed(run: Run) -> int:
    return (run.seed * 7919 + 13) % (2**32)


def list_runs(set_dir: Path) -> list[Path]:
    return sorted(d for d in Path(set_dir).iterdir() if (d / "frames.csv").exists())


# ---- augmentation: exact mirrors of the camera ---------------------------------------------

# Mirror M of the camera axes (x right, y down): the image flips, vectors map by M, rotation
# vectors (pseudo-vectors) by det(M)·M. Index 0 is the identity.
MIRRORS = [np.diag(m).astype(np.float64) for m in ([1, 1, 1], [-1, 1, 1], [1, -1, 1], [-1, -1, 1])]


def mirror_image(img: np.ndarray, k: int) -> np.ndarray:
    if k == 1:
        return img[..., :, ::-1]
    if k == 2:
        return img[..., ::-1, :]
    if k == 3:
        return img[..., ::-1, ::-1]
    return img


def mirror_vec(v: np.ndarray, k: int) -> np.ndarray:
    return v * np.diag(MIRRORS[k])


def mirror_rot(w: np.ndarray, k: int) -> np.ndarray:
    M = np.diag(MIRRORS[k])
    return w * M * np.prod(M)


def mirror_sensor(x: np.ndarray, k: int) -> np.ndarray:
    x = x.copy()
    x[0:3] = mirror_vec(x[0:3], k)
    x[3:6] = mirror_vec(x[3:6], k)
    x[6:9] = mirror_rot(x[6:9], k)
    x[10:13] = mirror_vec(x[10:13], k)
    return x


# ---- torch datasets -------------------------------------------------------------------------


class PairDataset:
    """Every consecutive pair of every run in a set; sensors drawn per sample (train) or per run (eval)."""

    def __init__(self, set_dir: Path, train: bool, noise: dict = NOISE, limit: int | None = None, images: bool = True, fx: bool = False,
                 airspeed: bool = True):
        self.images = images
        self.fx = fx
        self.airspeed = airspeed
        self.runs = [Run.load(d) for d in list_runs(set_dir)[:limit]]
        if not self.runs:
            raise ValueError(f"{set_dir}: no runs")
        self.cam = self.runs[0].cam
        odd = [r.id for r in self.runs if not r.cam.matches(self.cam)]
        if odd:
            raise ValueError(f"{set_dir}: {len(odd)} runs have another camera than {self.cam}, e.g. {odd[0]}")
        self.train = train
        self.noise = noise
        self.index = [(r, i) for r in range(len(self.runs)) for i in range(self.runs[r].n - 1)]
        self.labels = [run.labels() for run in self.runs]
        self._sensors: dict[int, Sensors] = {}
        self._fx: dict[int, Fx] = {}

    def __len__(self) -> int:
        return len(self.index)

    def _run_fx(self, r: int) -> Fx:
        if r not in self._fx:
            self._fx[r] = draw_fx(np.random.default_rng([run_seed(self.runs[r]), FX_STREAM]), (self.cam.h, self.cam.w))
        return self._fx[r]

    def sensors(self, r: int, rng: np.random.Generator | None = None) -> Sensors:
        if rng is not None:
            return simulate(self.runs[r], rng, self.noise)
        if r not in self._sensors:
            self._sensors[r] = simulate(self.runs[r], np.random.default_rng(run_seed(self.runs[r])), self.noise)
        return self._sensors[r]

    def __getitem__(self, j: int):
        r, i = self.index[j]
        run = self.runs[r]
        rng = np.random.default_rng() if self.train else None
        s = self.sensors(r, rng)
        img = np.stack([run.frames[i], run.frames[i + 1]]).astype(np.float32) / 255.0 if self.images else np.zeros((2, 1, 1), np.float32)
        if self.fx and self.images and not self.train:
            img = apply_fx(img, self._run_fx(r), np.random.default_rng([run_seed(run), FX_STREAM, i + 1]))
        x = sensor_vector(run, s, i)
        if not self.airspeed:
            x[10:14] = 0
        dp, rot = self.labels[r][0][i], self.labels[r][1][i]
        h = max(s.baro[i], 5.0)
        y_t = dp / h * 10                               # translation in tenths of a height
        y_r = (rot - s.gyro[i]) / D * 100               # gyro correction, centidegrees
        if self.train:
            k = int(rng.integers(4))
            img = mirror_image(img, k) if self.images else img
            x = mirror_sensor(x, k)
            y_t = mirror_vec(y_t, k)
            y_r = mirror_rot(y_r, k)
            if self.fx:
                img = apply_fx(img, draw_fx(rng, (self.cam.h, self.cam.w)), rng) if self.images else img
            else:
                gain = rng.uniform(0.8, 1.2)
                gamma = rng.uniform(0.8, 1.25)
                img = np.clip(img * gain * rng.uniform(0.98, 1.02, (2, 1, 1)), 0, 1) ** gamma
                img = img + rng.normal(0, 2 / 255, img.shape)
        y = np.concatenate([y_t, y_r]).astype(np.float32)
        return np.ascontiguousarray(img, dtype=np.float32), x, y, np.float32(h), r, i
