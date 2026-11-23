r"""Part IV, Section 5: the atom detector of a session.

Theory
======

The fitted atoms of a reconstruction, with their intensities, are labelled training
data for a network that finds atoms in the image (Part II, Section 8).  The
:class:`Detection` of a session

* builds the sample of the open reconstruction: target and second target
  (``i_target``, required) as input, all atoms with occupancy of at least one half
  inside the loss mask as label,
* trains a new detector or continues a saved one (so that several runs can be learnt
  one after the other), and keeps it under ``<workspace>/models/<name>`` (without a
  workspace under ``<out_dir>/models``),
* predicts the atoms of the open target and compares them with the fitted atoms:
  predicted atoms without a fitted partner are *new*, fitted atoms without a
  predicted partner are *unconfirmed*,
* inserts the new ones (with the predicted intensity, through the checks of the
  seeding) and writes the comparison as a table,
* freezes the open reconstruction as an entry of the collection of good
  reconstructions (Part II, Section 8.4), from which the tools train.

Implementation
==============
"""

from __future__ import annotations

import csv
import dataclasses
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..learning import detection as det
from ..physics.structure import atomic_number_of

MODELS_DIR = "models"
EXPORT_FILE, HEAT_FILE = "ml_detections.csv", "ml_heatmap.npy"
NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.\-]+$")
OCCUPIED = 0.5


class DetectionError(RuntimeError):
    pass


@dataclasses.dataclass
class Prediction:
    positions: np.ndarray        # (n, 2) row, col [px]
    intensity: np.ndarray        # (n,) predicted I
    heat: np.ndarray             # (H, W)
    model: str


class Detection:
    def __init__(self, session):
        self.session = session
        self.net: Optional[det.UNet] = None
        self.meta: Dict[str, Any] = {}
        self.name: Optional[str] = None
        self.trainer: Optional[det.DetectorTrainer] = None
        self.prediction: Optional[Prediction] = None
        #: the collection of good reconstructions (None: ``~/data/ml_collection``)
        self.collection_root: Optional[str] = None

    def reset(self) -> None:
        """A new reconstruction was opened: the prediction belonged to the old target."""

        self.prediction = None
        self.trainer = None

    # ------------------------------------------------------------------ where models live
    def root(self) -> str:
        s = self.session
        if s.workspace:
            return os.path.join(s.workspace, MODELS_DIR)
        if s.assembly is not None and s.assembly.out_dir:
            return os.path.join(s.assembly.out_dir, MODELS_DIR)
        raise DetectionError("models need a workspace (start the core with --workspace DIR) or an output folder")

    def models(self) -> List[Dict[str, Any]]:
        try:
            root = self.root()
        except DetectionError:
            return []
        if not os.path.isdir(root):
            return []
        out = []
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if det.is_detector(path):
                meta = det.read_meta(path)
                out.append({"name": name, "steps": meta.get("steps"), "runs": len(meta.get("sources", [])),
                            "val_f1": meta.get("val_f1"), "created": meta.get("created")})
        return out

    def _path(self, name: str) -> str:
        if not name or not NAME_PATTERN.match(name):
            raise DetectionError(f"invalid model name {name!r} (letters, digits, _ . - only)")
        return os.path.join(self.root(), name)

    def load(self, name: str) -> None:
        path = self._path(name)
        if not det.is_detector(path):
            raise DetectionError(f"no detector {name!r} in {self.root()}")
        self.net, self.meta = det.load_detector(path, self.session.device)
        self.name = name

    # ------------------------------------------------------------------ data of the open run
    def _inputs(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        s = self.session
        s.require_model()
        if s.alt_target is None:
            raise DetectionError("the detector needs the second target (i_target.tif next to the data)")
        _, loss_mask, _ = s.model.objective.regions(windowed=False)
        valid = None if loss_mask is None else loss_mask.detach().cpu().numpy() > 0
        return np.asarray(s.assembly.target, dtype=np.float32), np.asarray(s.alt_target, dtype=np.float32), valid

    def fitted_atoms(self) -> Tuple[np.ndarray, np.ndarray]:
        """Positions and intensities of all atoms with occupancy of at least one half."""

        model = self.session.require_model()
        positions, intensity = [], []
        with torch.no_grad():
            for i in range(len(model.experiment.layers)):
                keep = (model.params.occupancy(i) >= OCCUPIED).cpu().numpy()
                positions.append(model.params.positions(i).cpu().numpy()[keep])
                intensity.append(model.params.layers[i].intensity.detach().cpu().numpy()[keep])
        return (np.concatenate(positions).astype(np.float64).reshape(-1, 2),
                np.concatenate(intensity).astype(np.float64))

    def _inside(self, positions, intensity, valid=None):
        """The atoms that lie in the valid region (the loss mask)."""

        if valid is None:
            valid = self._inputs()[2]
        if valid is None or len(positions) == 0:
            return positions, intensity
        r = np.clip(np.round(positions[:, 0]).astype(int), 0, valid.shape[0] - 1)
        c = np.clip(np.round(positions[:, 1]).astype(int), 0, valid.shape[1] - 1)
        inside = valid[r, c]
        return positions[inside], intensity[inside]

    def pixel_A(self) -> float:
        return self.session.require_model().experiment.imaging.pixel_size_nm * 10.0

    # ------------------------------------------------------------------ collection
    def snapshot(self) -> Dict[str, Any]:
        """The open reconstruction as training data: images, fitted atoms in the valid region, pixel size."""

        target, second, valid = self._inputs()
        positions, intensity = self._inside(*self.fitted_atoms(), valid)
        if len(positions) == 0:
            raise DetectionError("no atoms to learn from (occupancy >= 0.5 inside the loss mask)")
        return {"target": target, "second": second, "valid": valid, "positions": positions, "intensity": intensity,
                "pixel_A": self.pixel_A(), "source": self.session.assembly.out_dir or ""}

    def collection(self) -> det.Collection:
        return det.Collection(self.collection_root)

    def add_to_collection(self, groups=(), name: str = "") -> Dict[str, Any]:
        """Freeze the open reconstruction as an entry of the collection."""

        snapshot = self.snapshot()
        name = name or os.path.basename(os.path.normpath(snapshot["source"])) or "lauf"
        entry_id = self.collection().add(snapshot, name, groups)
        return {"id": entry_id, "name": name, "atoms": int(len(snapshot["positions"])), "groups": list(groups)}

    # ------------------------------------------------------------------ training
    def start_training(self, name: str, resume: bool = False, sigma_A: float = 0.3, holdout: float = 0.15,
                       **config) -> det.DetectorTrainer:
        target, second, valid = self._inputs()
        self._path(name)                                   # checks the name before anything is computed
        positions, intensity = self._inside(*self.fitted_atoms(), valid)
        if len(positions) == 0:
            raise DetectionError("no atoms to learn from (occupancy >= 0.5 inside the loss mask)")
        net, meta = None, {}
        if resume:
            path = self._path(name)
            if not det.is_detector(path):
                raise DetectionError(f"cannot continue: no detector {name!r}")
            net, meta = det.load_detector(path, self.session.device)
        pixel_A = self.pixel_A()
        sigma_px = float(meta.get("sigma_px", max(1.0, float(sigma_A) / pixel_A)))
        scale = float(meta.get("scale", float(np.median(intensity)) or 1.0))
        sample = det.make_sample(target, second, positions, intensity, scale, sigma_px, valid, holdout,
                                 name=self.session.assembly.out_dir or "")
        known = {f.name for f in dataclasses.fields(det.DetectorConfig)}
        unknown = sorted(set(config) - known)
        if unknown:
            raise DetectionError(f"unknown detector options {unknown}")
        cfg = det.DetectorConfig(**config)
        if net is not None:
            cfg.base, cfg.depth = net.config["base"], net.config["depth"]
        tolerance_px, min_distance_px = det.validation_scales(positions, pixel_A)
        self.trainer = det.DetectorTrainer([sample], cfg, net, self.session.device, tolerance_px=tolerance_px,
                                           min_distance_px=min_distance_px)
        self.meta = {**meta, "sigma_px": sigma_px, "sigma_A": sigma_px * pixel_A, "scale": scale, "pixel_A": pixel_A,
                     "channels": ["target", "i_target"], "base": cfg.base, "depth": cfg.depth}
        if pixel_A and meta.get("pixel_A") and abs(meta["pixel_A"] - pixel_A) > 0.05 * pixel_A:
            self.session.assembly.notes.append(f"detector {name}: pixel size {pixel_A:.3g} A differs from "
                                               f"the training size {meta['pixel_A']:.3g} A")
        self.name = name
        self._training_atoms = len(positions)
        return self.trainer

    def finish_training(self) -> str:
        """Save the trained detector and predict with it."""

        t = self.trainer
        t.validate()
        self.net = t.net
        meta = self.meta
        meta["steps"] = int(meta.get("steps", 0)) + t.steps
        meta["sources"] = list(meta.get("sources", [])) + [{"run": self.session.assembly.out_dir,
                                                           "atoms": self._training_atoms, "steps": t.steps,
                                                           "val_loss": t.val_loss, "val_f1": t.val_f1}]
        meta.update({"val_loss": t.val_loss, "val_f1": t.val_f1, "loss": t.loss,
                     "created": time.strftime("%Y-%m-%d %H:%M:%S"), "history": t.history[-200:]})
        path = det.save_detector(self._path(self.name), self.net, meta)
        self.trainer = None
        self.predict()
        return path

    # ------------------------------------------------------------------ prediction
    def set_prediction(self, positions, intensity, model: str = "extern") -> Dict[str, Any]:
        """Take a prediction made elsewhere (the standalone predict interface) as the current one."""

        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
        intensity = np.asarray(intensity, dtype=np.float64).reshape(-1)
        shape = np.asarray(self.session.require_model() and self.session.assembly.target).shape
        self.prediction = Prediction(positions, intensity, np.zeros(shape, dtype=np.float32), model)
        return self.summary()

    def predict(self, name: Optional[str] = None, threshold: float = 0.3, min_distance_A: float = 0.8) -> Dict[str, Any]:
        if name and name != self.name:
            self.load(name)
        if self.net is None:
            raise DetectionError("no detector: train one or choose a saved one")
        target, second, valid = self._inputs()
        heat = det.predict(self.net, det.inputs(target, second, valid), self.session.device)
        min_px = max(1.0, float(min_distance_A) / self.pixel_A())
        positions, heights = det.find_peaks(heat, float(threshold), min_px, valid)
        scale = float(self.meta.get("scale", 1.0))
        self.prediction = Prediction(positions, heights * scale, heat * scale, self.name or "")
        return self.summary()

    def compare(self, tolerance_A: float = 0.5):
        """``(prediction new mask, unconfirmed fitted positions, fitted intensity of those)``."""

        p = self.prediction
        if p is None:
            raise DetectionError("nothing predicted yet")
        fitted, intensity = self._inside(*self.fitted_atoms())
        tolerance_px = float(tolerance_A) / self.pixel_A()
        matched, confirmed = det.match(p.positions, fitted, tolerance_px)
        return ~matched, fitted[~confirmed], intensity[~confirmed]

    def summary(self, tolerance_A: float = 0.5) -> Dict[str, Any]:
        new, unconfirmed, _ = self.compare(tolerance_A)
        p = self.prediction
        return {"model": p.model, "predicted": int(len(p.positions)), "new": int(new.sum()),
                "unconfirmed": int(len(unconfirmed)), "val_f1": self.meta.get("val_f1")}

    def result(self, tolerance_A: float = 0.5) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        new, unconfirmed, _ = self.compare(tolerance_A)
        p = self.prediction
        arrays = {"pred.pos": p.positions.astype(np.float32), "pred.int": p.intensity.astype(np.float32),
                  "pred.new": new.astype(np.uint8), "fit.unconfirmed": unconfirmed.astype(np.float32)}
        return self.summary(tolerance_A), arrays

    # ------------------------------------------------------------------ using it
    def insert(self, layer: Any = None, tolerance_A: float = 0.5, check_collisions: bool = True,
               check_loss: bool = True) -> Dict[str, Any]:
        """Insert the new predicted atoms into ``layer``, brightest first, with their predicted intensity."""

        new, _, _ = self.compare(tolerance_A)
        p = self.prediction
        points, intensity = p.positions[new], p.intensity[new]
        s = self.session
        seeder = s.make_seeder(layer)
        z = float(atomic_number_of(seeder.store.spec.element))
        options = np.clip(intensity / z, 0.05, 1.5)
        before = seeder.n_atoms
        accepted = seeder.filter_points(points, commit=True, check_margin=True, check_collisions=check_collisions,
                                        check_loss=check_loss, options=options)
        s.revision += 1
        return {"candidates": int(len(points)), "accepted": int(len(accepted)), "atoms_before": before,
                "atoms": seeder.n_atoms, "report": dict(seeder.filter_report)}

    def export(self, tolerance_A: float = 0.5) -> str:
        """``ml_detections.csv`` (row, col, I, status) and ``ml_heatmap.npy`` in the output folder."""

        new, unconfirmed, unconfirmed_I = self.compare(tolerance_A)
        p, s = self.prediction, self.session
        folder = (s.assembly.out_dir if s.assembly is not None and s.assembly.out_dir else None) or s.last_save
        if not folder:
            raise DetectionError("no output folder to write to")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, EXPORT_FILE)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["row_px", "col_px", "intensity", "status", "model"])
            for (row, col), value, is_new in zip(p.positions, p.intensity, new):
                writer.writerow([f"{row:.3f}", f"{col:.3f}", f"{value:.4g}", "new" if is_new else "matched", p.model])
            for (row, col), value in zip(unconfirmed, unconfirmed_I):
                writer.writerow([f"{row:.3f}", f"{col:.3f}", f"{value:.4g}", "unconfirmed", p.model])
        np.save(os.path.join(folder, HEAT_FILE), p.heat.astype(np.float32))
        return path
