r"""Part V, Section 14: a batch job -- what is processed, how, and where it stands.

A job is a folder (the *output folder*) with

* ``job.json``   -- the inputs, the shared template, the pipeline, the device;
* ``status.json`` -- written by the runner: state, current input/frame/step, counts;
* ``log.txt``    -- one line per event; ``runner.out`` -- the runner's own output;
* ``STOP``       -- created by the window to ask the runner to stop;
* ``<input>/``   -- the results of each input (see :mod:`temsim_ml.batch.runner`).

An **input** is a data folder (``target``, ``i_target``, mask, ``init.json``) or a pair
of files (target and i_target, optionally a mask).  Either image may be a *stack*
(a multi-page TIF): then every frame is reconstructed, and frame :math:`n+1` may
start from frame :math:`n` (``warm_start``: ``none``, ``optics``, ``all``).

The **pipeline** is a list of steps ``{"kind": ..., "params": {...}}``; the kinds and
their parameters are described by :data:`STEP_TYPES` (from which the window builds
its forms).

Implementation
==============
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

JOB_FILE = "job.json"
STATUS_FILE = "status.json"
LOG_FILE = "log.txt"
STOP_FILE = "STOP"
REPORT_FILE = "report.json"
ATOMS_FILE = "atoms.csv"
PIPELINES = os.path.expanduser(os.environ.get("TEMSIM_PIPELINES") or "~/data/batch_pipelines")

WARM_STARTS = {"none": "nichts", "optics": "Aberrationen", "all": "alles (Atome + Optik)"}
ABERRATION_CHOICES = {"all": "alle", "C10,C30": "C10 + C30", "C10,C12,C30": "C10 + C12 + C30", "none": "keine"}

#: Step kinds: label and parameters ``(key, label, type, default, extra)``; type is
#: ``bool``, ``int``, ``float``, ``choice`` (extra: {value: label}) or ``model``.
STEP_TYPES: Dict[str, Dict[str, Any]] = {
    "ml": {"label": "ML-Vorhersage + Einfügen", "params": [
        ("model", "Modell", "model", "", "leer: das neueste globale Modell"),
        ("threshold", "Schwelle", "float", 0.3, (0.0, 1.0, 0.05, 2)),
        ("min_distance_A", "Mindestabstand (Å)", "float", 0.8, (0.1, 10.0, 0.1, 2)),
        ("tolerance_A", "Toleranz neu/bekannt (Å)", "float", 0.5, (0.05, 5.0, 0.05, 2)),
        ("insert", "neue Atome einfügen", "bool", True, None),
        ("check_loss", "nur wenn Loss sinkt", "bool", False, None),
        ("check_collisions", "Kollisionen prüfen", "bool", True, None),
        ("remove_unconfirmed", "unbestätigte Atome entfernen", "bool", False, None),
    ]},
    "train": {"label": "Training", "params": [
        ("max_steps", "max. Schritte", "int", 2000, (1, 1_000_000, 100)),
        ("patience", "Konvergenz: Fenster (Schritte, 0 = aus)", "int", 200, (0, 1_000_000, 50)),
        ("rel_tol", "Konvergenz: min. rel. Verbesserung", "float", 1e-4, (0.0, 0.1, 1e-4, 6)),
        ("aberrations", "Aberrationen", "choice", "all", ABERRATION_CHOICES),
        ("positions", "Positionen", "bool", True, None),
        ("intensities", "Intensitäten", "bool", True, None),
        ("occupancies", "Besetzungen", "bool", False, None),
        ("normalisation", "Normierung", "bool", True, None),
        ("lr_aberrations", "LR Aberrationen", "float", 1e-2, (0.0, 1.0, 1e-3, 6)),
        ("lr_positions", "LR Positionen", "float", 1e-3, (0.0, 1.0, 1e-4, 6)),
        ("lr_intensities", "LR Intensitäten", "float", 1e-4, (0.0, 1.0, 1e-5, 6)),
        ("repulsion", "Abstoßung", "float", 10.0, (0.0, 1000.0, 1.0, 2)),
        ("repulsion_A", "Abstoßung ab (Å)", "float", 1.42, (0.0, 10.0, 0.05, 2)),
    ]},
    "seeding": {"label": "Seeding", "params": [
        ("trials", "Versuche", "int", 2000, (1, 10_000_000, 100)),
        ("tune_every", "Optik nachführen alle (Versuche, 0 = nie)", "int", 0, (0, 1_000_000, 50)),
        ("tune_steps", "Schritte je Nachführung", "int", 0, (0, 10_000, 5)),
    ]},
    "propeller": {"label": "Propeller", "params": [
        ("distance_A", "Abstand (Å)", "float", 0.5, (0.05, 5.0, 0.05, 2)),
        ("angles", "Winkel", "int", 12, (1, 360, 1)),
        ("min_gain", "min. Gewinn", "float", 0.0, (0.0, 1.0, 1e-4, 6)),
        ("first", "beim ersten Treffer beenden", "bool", False, None),
    ]},
}


class JobError(ValueError):
    pass


def step_defaults(kind: str) -> Dict[str, Any]:
    if kind not in STEP_TYPES:
        raise JobError(f"unknown step {kind!r}; known: {sorted(STEP_TYPES)}")
    return {key: default for key, _, _, default, _ in STEP_TYPES[kind]["params"]}


def make_step(kind: str, **params) -> Dict[str, Any]:
    defaults = step_defaults(kind)
    unknown = sorted(set(params) - set(defaults))
    if unknown:
        raise JobError(f"unknown parameters of {kind}: {unknown}")
    return {"kind": kind, "params": {**defaults, **params}}


def default_pipeline() -> List[Dict[str, Any]]:
    """ML prediction with the global model, then training of aberrations, positions and intensities."""

    return [make_step("ml"), make_step("train")]


def step_title(step: Dict[str, Any]) -> str:
    kind, p = step["kind"], step.get("params", {})
    label = STEP_TYPES[kind]["label"]
    if kind == "ml":
        return f"{label} ({p.get('model') or 'global, neuestes'})"
    if kind == "train":
        parts = [ABERRATION_CHOICES.get(p.get("aberrations"), "?") + " Aberr."]
        parts += [name for key, name in (("positions", "Pos."), ("intensities", "I"), ("occupancies", "Bes."))
                  if p.get(key)]
        return f"{label}: {', '.join(parts)}, ≤{p.get('max_steps')} Schritte"
    if kind == "seeding":
        return f"{label}: {p.get('trials')} Versuche"
    return label


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

TARGET_NAMES = ("target.tif", "target.tiff", "target.npy")
I_TARGET_NAMES = ("i_target.tif", "i_target.tiff", "i_target.npy")
INIT_NAMES = ("init.json", "init_random.json")


def _first(folder: str, names) -> str:
    for name in names:
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    return ""


def folder_input(folder: str) -> Dict[str, Any]:
    """An input from a data folder in the usual layout."""

    from temsim.data.dataset import find_masks
    folder = os.path.abspath(folder)
    target = _first(folder, TARGET_NAMES)
    if not target:
        raise JobError(f"{folder}: kein target.tif/.npy")
    masks = find_masks(folder)
    return new_input(os.path.basename(folder.rstrip("/")), target, _first(folder, I_TARGET_NAMES),
                     masks[0] if masks else "", _first(folder, INIT_NAMES), kind="folder", folder=folder)


def file_input(target: str, i_target: str, mask: str = "") -> Dict[str, Any]:
    name = os.path.splitext(os.path.basename(target))[0]
    return new_input(name, os.path.abspath(target), os.path.abspath(i_target) if i_target else "",
                     os.path.abspath(mask) if mask else "", "", kind="files")


def new_input(name, target, i_target, mask, init, kind="files", folder="") -> Dict[str, Any]:
    entry = {"name": name, "kind": kind, "folder": folder, "target": target, "i_target": i_target, "mask": mask,
             "init": init, "pixel_size_nm": None, "voltage_kV": None, "mode": None, "warm_start": "optics"}
    entry["frames"] = image_frames(target)
    return entry


def image_frames(path: str) -> int:
    """Number of frames of an image file (1 for a plain image) without reading the pixels."""

    if not path:
        return 0
    if path.lower().endswith(".npy"):
        shape = np.load(path, mmap_mode="r").shape
        return int(shape[0]) if len(shape) == 3 else 1
    import tifffile
    with tifffile.TiffFile(path) as tif:
        shape = tif.series[0].shape
    shape = [s for s in shape if s != 1] or [1]
    return int(np.prod(shape[:-2])) if len(shape) > 2 else 1


def read_frames(path: str) -> np.ndarray:
    """``(n, H, W)`` float32 (a plain image gives ``n = 1``)."""

    from temsim.data.dataset import read_image
    image = np.asarray(read_image(path))
    image = np.squeeze(image)
    if image.ndim == 2:
        image = image[None]
    if image.ndim != 3:
        raise JobError(f"{path}: {image.shape} is neither an image nor a stack")
    return image.astype(np.float32)


def read_init(entry: Dict[str, Any], template: Dict[str, Any]) -> Dict[str, Any]:
    """The ``init.json`` of an input: its own file or the template's.  A value given for the input always wins;
    a value of the template only fills what the file does not say."""

    init: Dict[str, Any] = {}
    path = entry.get("init") or template.get("init") or ""
    if path:
        with open(path, "r", encoding="utf-8") as f:
            init = json.load(f)
    init.pop("target_file", None)
    init.pop("shape", None)
    imaging = init.setdefault("imaging", {})
    current = {"pixel_size_nm": init.get("pixel_size_nm"), "voltage_kV": init.get("voltage_kV"),
               "mode": imaging.get("mode")}
    for key in current:
        value = entry.get(key) or (None if current[key] else template.get(key))
        if not value:
            continue
        if key == "mode":
            imaging["mode"] = str(value)
        else:
            init[key] = float(value)
    if not init.get("pixel_size_nm"):
        raise JobError(f"{entry['name']}: Pixelgröße unbekannt (init.json oder Eingabe in der Liste)")
    return init


def check_job(job: Dict[str, Any]) -> List[str]:
    """Problems that would stop the job (empty list: fine)."""

    problems = []
    if not job.get("inputs"):
        problems.append("keine Eingaben")
    if not job.get("pipeline"):
        problems.append("die Pipeline ist leer")
    if not job.get("output"):
        problems.append("kein Ausgabeordner")
    names = [e["name"] for e in job.get("inputs", [])]
    if len(set(names)) != len(names):
        problems.append("Eingaben haben doppelte Namen")
    needs_second = any(s["kind"] == "ml" for s in job.get("pipeline", []))
    for e in job.get("inputs", []):
        if not os.path.exists(e["target"]):
            problems.append(f"{e['name']}: target fehlt ({e['target']})")
            continue
        if needs_second:
            if not e.get("i_target") or not os.path.exists(e["i_target"]):
                problems.append(f"{e['name']}: kein i_target (die ML-Vorhersage braucht es)")
            else:
                n = image_frames(e["i_target"])
                if n not in (1, e["frames"]):
                    problems.append(f"{e['name']}: i_target hat {n} Frames, target {e['frames']}")
        try:
            read_init(e, job.get("template", {}))
        except (JobError, OSError, ValueError) as exc:
            problems.append(str(exc))
    for step in job.get("pipeline", []):
        if step.get("kind") not in STEP_TYPES:
            problems.append(f"unbekannter Schritt {step.get('kind')!r}")
    return problems


def safe_name(name: str) -> str:
    keep = "".join(c if c.isalnum() or c in "._-" else "_" for c in name).strip("._")
    return keep or "eingabe"


def input_dir(job_dir: str, entry: Dict[str, Any]) -> str:
    return os.path.join(job_dir, safe_name(entry["name"]))


def frame_dir(job_dir: str, entry: Dict[str, Any], frame: int) -> str:
    """Single images: the input's folder itself; stacks: ``frame_000`` ... inside it."""

    base = input_dir(job_dir, entry)
    return base if entry["frames"] <= 1 else os.path.join(base, f"frame_{frame:03d}")


# ---------------------------------------------------------------------------
# files of a job
# ---------------------------------------------------------------------------

def write_json(path: str, data: Any) -> None:
    partial = path + ".partial"
    with open(partial, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, default=_plain)
    os.replace(partial, path)


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def read_json(path: str, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def save_job(job: Dict[str, Any]) -> str:
    os.makedirs(job["output"], exist_ok=True)
    job = {**job, "version": 1, "saved": time.strftime("%Y-%m-%d %H:%M:%S")}
    path = os.path.join(job["output"], JOB_FILE)
    write_json(path, job)
    return path


def load_job(job_dir: str) -> Dict[str, Any]:
    job = read_json(os.path.join(job_dir, JOB_FILE))
    if job is None:
        raise JobError(f"kein Auftrag in {job_dir}")
    job["output"] = os.path.abspath(job_dir)
    return job


def frame_done(job_dir: str, entry: Dict[str, Any], frame: int) -> bool:
    report = read_json(os.path.join(frame_dir(job_dir, entry, frame), REPORT_FILE), {})
    return bool(report.get("ok"))


def pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:                                                     # a zombie (finished, not yet reaped) is not alive
        with open(f"/proc/{int(pid)}/stat") as f:
            return f.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return True


def status(job_dir: str) -> Dict[str, Any]:
    """The runner's status; ``running`` only while its process lives."""

    state = read_json(os.path.join(job_dir, STATUS_FILE), {}) or {}
    if state.get("state") == "running" and not pid_alive(state.get("pid")):
        state["state"] = "abgebrochen"
    return state


def is_running(job_dir: str) -> bool:
    return status(job_dir).get("state") == "running"


def list_pipelines(directory: str = PIPELINES) -> List[str]:
    if not os.path.isdir(directory):
        return []
    return sorted(os.path.splitext(n)[0] for n in os.listdir(directory) if n.endswith(".json"))


def save_pipeline(name: str, pipeline: List[Dict[str, Any]], directory: str = PIPELINES) -> str:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, safe_name(name) + ".json")
    write_json(path, pipeline)
    return path


def load_pipeline(name: str, directory: str = PIPELINES) -> List[Dict[str, Any]]:
    pipeline = read_json(os.path.join(directory, safe_name(name) + ".json"))
    if not isinstance(pipeline, list):
        raise JobError(f"Vorlage {name!r} nicht lesbar")
    return normalize_pipeline(pipeline)


def normalize_pipeline(pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Steps with every parameter (missing ones at their defaults, unknown ones dropped)."""

    return [make_step(s["kind"], **{k: v for k, v in s.get("params", {}).items() if k in step_defaults(s["kind"])})
            for s in pipeline]


def tail(path: str, lines: int = 200) -> Tuple[str, int]:
    """The last ``lines`` lines of a text file and its size."""

    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 64_000))
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return "", 0
    return "\n".join(text.splitlines()[-lines:]), size
