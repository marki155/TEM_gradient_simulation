r"""Part V, Section 14: the batch runner -- a job without anybody watching.

``python -m temsim_ml.batch run JOBDIR`` works through ``JOBDIR/job.json``: every
frame of every input is opened as a fresh reconstruction (the seeding route of
Part IV, Section 3: one empty layer, the microscope from ``init.json``), then the
steps of the pipeline run in order:

* ``ml``        -- the detector predicts atoms from target and i_target; unconfirmed
  fitted atoms may be removed, new ones are inserted with their predicted
  intensity (optionally only where the loss falls);
* ``train``     -- the chosen quantities are released (refinement settings with
  repulsion) and optimised until ``max_steps`` or until the best loss has not
  improved by ``rel_tol`` (relative) within ``patience`` steps;
* ``seeding``, ``propeller`` -- the procedures of Part II, Section 6.

**Stacks.**  Frame :math:`n+1` may start from frame :math:`n`: ``optics`` copies
aberrations and normalisation, ``all`` the whole model (atoms included).  After a
failed frame the next one starts afresh.

**Output** per frame (single images: in ``JOBDIR/<input>/``, stacks: in
``JOBDIR/<input>/frame_NNN/``): ``run/`` (the run directory as the main interface
saves it), ``ml_detections.csv`` and ``ml_heatmap.npy`` (after an ML step),
``atoms.csv``, ``loss.csv`` and ``report.json``.  Per input the atom pictures of
:mod:`temsim_ml.batch.render`.

**Robustness.**  An error in a frame is logged and the next frame follows.  A frame
counts as done when its ``report.json`` says so; starting the job again skips
done frames.  The file ``STOP`` ends the job after the current unit of work.

Implementation
==============
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
import traceback
from dataclasses import replace
from typing import Any, Dict, List, Optional

import numpy as np

from . import job as jobs
from . import render


class Stopped(Exception):
    pass


class Runner:
    def __init__(self, job_dir: str, device: Optional[str] = None):
        self.job = jobs.load_job(job_dir)
        self.dir = self.job["output"]
        self.device = device or self.job.get("device") or "auto"
        self.session = None
        self._models: Dict[str, Any] = {}
        self.state: Dict[str, Any] = {}

    # ------------------------------------------------------------------ bookkeeping
    def log(self, message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {message}"
        with open(os.path.join(self.dir, jobs.LOG_FILE), "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True)

    def publish(self, **changes) -> None:
        self.state.update(changes, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
        jobs.write_json(os.path.join(self.dir, jobs.STATUS_FILE), self.state)

    def check_stop(self) -> None:
        if os.path.exists(os.path.join(self.dir, jobs.STOP_FILE)):
            raise Stopped()

    # ------------------------------------------------------------------ the job
    def run(self) -> str:
        from temsim.service.session import Session

        if jobs.is_running(self.dir) and jobs.status(self.dir).get("pid") != os.getpid():
            raise jobs.JobError("in diesem Ordner läuft schon ein Auftrag")
        try:
            os.remove(os.path.join(self.dir, jobs.STOP_FILE))
        except FileNotFoundError:
            pass
        inputs = self.job["inputs"]
        total = sum(int(e["frames"]) for e in inputs)
        done = sum(jobs.frame_done(self.dir, e, k) for e in inputs for k in range(int(e["frames"])))
        self.state = {"state": "running", "pid": os.getpid(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                      "total": total, "done": done, "failed": 0, "current": {}, "inputs": {}}
        self.session = Session(self.device)
        self.publish(device=self.session.device)
        self.log(f"Auftrag startet: {len(inputs)} Eingaben, {total} Bilder ({done} schon fertig), "
                 f"Gerät {self.session.device}")
        outcome = "finished"
        try:
            for entry in inputs:
                self.run_input(entry)
        except Stopped:
            outcome = "stopped"
            self.log("gestoppt")
        except Exception as exc:                         # noqa: BLE001 -- nothing may end the night silently
            outcome = "failed"
            self.log(f"Auftrag abgebrochen: {exc}\n{traceback.format_exc()}")
        self.publish(state=outcome, current={})
        if outcome == "finished":
            self.log(f"fertig: {self.state['done']} von {total} Bildern, {self.state['failed']} Fehler")
        return outcome

    def run_input(self, entry: Dict[str, Any]) -> None:
        from temsim.data.dataset import prepare_target, read_image

        n = int(entry["frames"])
        folder = jobs.input_dir(self.dir, entry)
        os.makedirs(folder, exist_ok=True)
        targets = jobs.read_frames(entry["target"])
        seconds = jobs.read_frames(entry["i_target"]) if entry.get("i_target") else None
        if len(targets) != n:
            n = entry["frames"] = len(targets)
        mask = None
        if entry.get("mask"):
            mask = np.squeeze(np.asarray(read_image(entry["mask"]))) > 0
        init = jobs.read_init(entry, self.job.get("template", {}))
        self.state["inputs"][entry["name"]] = {"frames": n, "done": 0, "failed": 0}
        self.log(f"Eingabe {entry['name']}: {n} Bild(er)")
        previous = None
        for k in range(n):
            self.check_stop()
            where = jobs.frame_dir(self.dir, entry, k)
            info = self.state["inputs"][entry["name"]]
            if jobs.frame_done(self.dir, entry, k):
                info["done"] += 1
                previous = "saved"
                continue
            second = None if seconds is None else seconds[k if len(seconds) > 1 else 0]
            try:
                self.publish(current={"input": entry["name"], "frame": k, "frames": n})
                previous = self.run_frame(entry, k, where, prepare_target(targets[k]), second, mask, init, previous)
                info["done"] += 1
                self.publish(done=self.state["done"] + 1)
            except Stopped:
                raise
            except Exception as exc:                     # noqa: BLE001
                previous = None
                info["failed"] += 1
                self.publish(failed=self.state["failed"] + 1)
                self.log(f"  Frame {k}: FEHLER {exc}")
                os.makedirs(where, exist_ok=True)
                jobs.write_json(os.path.join(where, jobs.REPORT_FILE),
                                {"ok": False, "error": str(exc), "traceback": traceback.format_exc()})
            finally:
                self._free()
        self.pictures(entry, folder, targets, init)

    def pictures(self, entry, folder, targets, init) -> None:
        atoms = []
        for k in range(len(targets)):
            atoms.append(render.read_atoms(os.path.join(jobs.frame_dir(self.dir, entry, k), jobs.ATOMS_FILE)))
        try:
            render.render_input(folder, targets, atoms, float(init["pixel_size_nm"]) * 10.0)
        except Exception as exc:                         # noqa: BLE001
            self.log(f"  Bilder von {entry['name']}: FEHLER {exc}")

    def _free(self) -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:                                # noqa: BLE001
            pass

    # ------------------------------------------------------------------ one frame
    def open_frame(self, entry, k, where, target, second, mask, init, previous):
        from temsim.data import store
        from temsim.data.dataset import Dataset, group_name_from_mask, prepare_target, spec_from_init
        from temsim.service import assemble

        s = self.session
        spec = spec_from_init(init)
        masks = {}
        if mask is not None:
            if mask.shape != target.shape:
                raise jobs.JobError(f"Maske {mask.shape} passt nicht zum Bild {target.shape}")
            masks[spec.groups[0].name if spec.groups else group_name_from_mask(entry["mask"])] = mask
        data = Dataset(folder=where, spec=spec, target=target, masks=masks)
        built = replace(assemble.seeding_assembly(data), out_dir=where)
        mode = entry.get("warm_start", "none") if int(entry["frames"]) > 1 else "none"
        if previous == "saved" and mode != "none" and k > 0:
            try:
                run = store.load_run(os.path.join(jobs.frame_dir(self.dir, entry, k - 1), "run"))
                previous = {"experiment": run.experiment, "settings": run.train, "state": run.state}
            except Exception as exc:                     # noqa: BLE001
                self.log(f"  Frame {k}: Vorgänger nicht lesbar ({exc}), Neustart")
                previous = None
        if not isinstance(previous, dict):
            previous = None
        if previous is not None and mode == "all":
            built = replace(built, experiment=previous["experiment"], settings=copy.deepcopy(previous["settings"]),
                            state=previous["state"], notes=built.notes + ["Start: Modell des vorigen Frames"])
        s.open_assembly(built)
        s.alt_target = None if second is None else prepare_target(second)
        if second is not None and s.alt_target.shape != target.shape:
            raise jobs.JobError(f"i_target {s.alt_target.shape} passt nicht zum Bild {target.shape}")
        if previous is not None and mode == "optics" and previous["state"].optics is not None:
            self._copy_optics(previous["state"])
            s.assembly.notes.append("Start: Optik des vorigen Frames")
        return mode

    def _copy_optics(self, state) -> None:
        import torch
        p = self.session.model.params
        with torch.no_grad():
            for key, value in (state.optics.aberrations or {}).items():
                if key in p.aberrations:
                    p.aberrations[key].fill_(float(value))
            if state.normalisation is not None:
                p.normalisation.copy_(torch.as_tensor(np.asarray(state.normalisation, dtype=np.float32),
                                                      device=p.normalisation.device).reshape(p.normalisation.shape))
        self.session.settings_changed()

    def run_frame(self, entry, k, where, target, second, mask, init, previous):
        s = self.session
        started = time.monotonic()
        os.makedirs(where, exist_ok=True)
        for name in (jobs.REPORT_FILE,):
            try:
                os.remove(os.path.join(where, name))
            except FileNotFoundError:
                pass
        mode = self.open_frame(entry, k, where, target, second, mask, init, previous)
        tag = f"{entry['name']}" + (f" Frame {k}" if int(entry["frames"]) > 1 else "")
        self.log(f"{tag}: Start ({'aus Vorgänger: ' + jobs.WARM_STARTS[mode] if mode != 'none' and k else 'neu'})")
        steps = []
        for index, step in enumerate(self.job["pipeline"]):
            self.check_stop()
            self.publish(current={**self.state["current"], "step": index, "steps": len(self.job["pipeline"]),
                                  "kind": step["kind"], "title": jobs.step_title(step), "progress": None})
            t0 = time.monotonic()
            result = getattr(self, "step_" + step["kind"])(dict(step.get("params", {})))
            result.update(kind=step["kind"], seconds=round(time.monotonic() - t0, 1))
            steps.append(result)
            self.log(f"  {jobs.step_title(step)}: " + ", ".join(f"{a}={b}" for a, b in result.items()
                                                                 if a not in ("kind",)))
        s.save(directory=os.path.join(where, "run"))
        positions, intensity, layers = self._atoms()
        render.write_atoms(os.path.join(where, jobs.ATOMS_FILE), positions, intensity, layers)
        if s.detection.prediction is not None:
            try:
                s.detection.export()
            except Exception as exc:                     # noqa: BLE001
                self.log(f"  ML-Export: {exc}")
        history = s.trainer.history if s.trainer else []
        with open(os.path.join(where, "loss.csv"), "w") as f:
            f.write("step,loss\n" + "".join(f"{i + 1},{v:.6g}\n" for i, v in enumerate(history)))
        loss = float(s.model.evaluate()["loss"])
        report = {"ok": True, "input": entry["name"], "frame": k, "warm_start": mode, "atoms": int(len(positions)),
                  "loss": loss, "seconds": round(time.monotonic() - started, 1), "steps": steps,
                  "optics": {key: v for key, v in s.values().items() if key.startswith("optics.")},
                  "notes": list(s.assembly.notes)}
        jobs.write_json(os.path.join(where, jobs.REPORT_FILE), report)
        self.log(f"{tag}: fertig, {len(positions)} Atome, Loss {loss:.4g}, {report['seconds']} s")
        return {"experiment": s.experiment(), "settings": copy.deepcopy(s.model.settings), "state": s.model.state()}

    def _atoms(self):
        import torch
        from temsim.service.detection import OCCUPIED
        model = self.session.model
        positions, intensity, layers = [], [], []
        with torch.no_grad():
            for i in range(len(model.experiment.layers)):
                keep = (model.params.occupancy(i) >= OCCUPIED).cpu().numpy()
                positions.append(model.params.positions(i).cpu().numpy()[keep])
                intensity.append(model.params.layers[i].intensity.detach().cpu().numpy()[keep])
                layers.append(np.full(int(keep.sum()), i))
        return (np.concatenate(positions).reshape(-1, 2), np.concatenate(intensity), np.concatenate(layers))

    # ------------------------------------------------------------------ steps
    def _detector(self, name: str):
        from temsim.learning import detection as det
        from temsim_ml.models import GLOBAL_MODELS, list_models

        if not name:
            found = list_models(GLOBAL_MODELS)
            if not found:
                raise jobs.JobError(f"kein globales Modell in {GLOBAL_MODELS}")
            path = found[0]["path"]
        elif os.path.isabs(name) or os.sep in name:
            path = os.path.expanduser(name)
        else:
            path = os.path.join(GLOBAL_MODELS, name)
        if not det.is_detector(path):
            raise jobs.JobError(f"kein Detektor in {path}")
        if path not in self._models:
            self._models[path] = det.load_detector(path, self.session.device)
        return path, self._models[path]

    def step_ml(self, p: Dict[str, Any]) -> Dict[str, Any]:
        s = self.session
        d = s.detection
        path, (net, meta) = self._detector(p.get("model", ""))
        d.net, d.meta, d.name = net, dict(meta), os.path.basename(path.rstrip("/"))
        summary = d.predict(threshold=float(p["threshold"]), min_distance_A=float(p["min_distance_A"]))
        tolerance = float(p["tolerance_A"])
        out = {"model": d.name, "predicted": summary["predicted"]}
        if p.get("remove_unconfirmed"):
            _, unconfirmed, _ = d.compare(tolerance)
            if len(unconfirmed):
                removed = 0
                for i in range(len(s.model.experiment.layers)):
                    removed += s.edit("remove_near", points=unconfirmed.tolist(),
                                      radius_px=tolerance / d.pixel_A(), layer=i)["removed"]
                out["removed"] = removed
        if p.get("insert", True):
            report = d.insert(tolerance_A=tolerance, check_collisions=bool(p.get("check_collisions", True)),
                              check_loss=bool(p.get("check_loss", False)))
            out.update(new=report["candidates"], inserted=report["accepted"])
        out["atoms"] = int(sum(s.model.params.counts()))
        return out

    def step_train(self, p: Dict[str, Any]) -> Dict[str, Any]:
        from temsim.service.jobs import TrainingJob

        s = self.session
        model = s.model
        if sum(model.params.counts()) == 0 and (p.get("positions") or p.get("intensities")):
            self.log("  Training: keine Atome im Modell")
        s.refine(repulsion=float(p["repulsion"]), repulsion_A=float(p["repulsion_A"]),
                 train_occupancies=bool(p.get("occupancies")))
        st = model.settings
        choice = p.get("aberrations", "all")
        wanted = set(st.aberration_flags) if choice == "all" else set() if choice == "none" else set(choice.split(","))
        st.aberration_flags = {key: key in wanted for key in st.aberration_flags}
        st.aberrations.enabled = bool(wanted)
        st.aberrations.lr = float(p["lr_aberrations"])
        st.normalisation.enabled = bool(p.get("normalisation", True))
        name = model.experiment.layers[s.layer_index()].name
        lt = st.layer(name)
        lt.free_positions.enabled, lt.free_positions.lr = bool(p.get("positions")), float(p["lr_positions"])
        lt.intensities.enabled, lt.intensities.lr = bool(p.get("intensities")), float(p["lr_intensities"])
        lt.occupancies.enabled = bool(p.get("occupancies"))
        s.settings_changed()
        job = TrainingJob(s, steps=int(p["max_steps"]))
        patience, rel_tol = int(p.get("patience", 0)), float(p.get("rel_tol", 0.0))
        best, best_step, reason = np.inf, 0, "max. Schritte"
        last_publish = 0.0
        while not job.done:
            status = job.advance()
            loss = status["loss"]
            if not np.isfinite(loss):
                raise RuntimeError(f"Loss ist {loss} nach {job.count} Schritten")
            if loss < best - rel_tol * abs(best if np.isfinite(best) else loss):
                best, best_step = loss, job.count
            if patience and job.count - best_step >= patience:
                reason = "konvergiert"
                break
            if time.monotonic() - last_publish > 2.0:
                self.check_stop()
                self.publish(current={**self.state["current"], "progress": [job.count, job.limit],
                                      "loss": loss})
                last_publish = time.monotonic()
        merged = job.merged
        return {"steps": job.count, "loss": round(float(job.last.get("loss", np.nan)), 6), "stop": reason,
                **({"merged": merged} if merged else {})}

    def step_seeding(self, p: Dict[str, Any]) -> Dict[str, Any]:
        from temsim.service.jobs import SeedingJob
        job = SeedingJob(self.session, trials=int(p["trials"]), tune_every=int(p["tune_every"]),
                         tune_steps=int(p["tune_steps"]))
        status: Dict[str, Any] = {}
        while not job.done:
            self.check_stop()
            status = job.advance()
            self.publish(current={**self.state["current"], "progress": [job.count, job.limit]})
        return {"trials": job.count, "atoms": status.get("atoms"), "accepted": status.get("accepted")}

    def step_propeller(self, p: Dict[str, Any]) -> Dict[str, Any]:
        from temsim.service.jobs import PropellerJob
        job = PropellerJob(self.session, distance_A=float(p["distance_A"]), angles=int(p["angles"]),
                           min_gain=float(p["min_gain"]), first=bool(p["first"]))
        status: Dict[str, Any] = {}
        while not job.done:
            self.check_stop()
            status = job.advance()
            self.publish(current={**self.state["current"], "progress": [job.count, job.total]})
        return {"split": status.get("split", 0), "atoms": status.get("atoms")}


# ---------------------------------------------------------------------------
# starting
# ---------------------------------------------------------------------------

def run_job(job_dir: str, device: Optional[str] = None) -> str:
    return Runner(job_dir, device).run()


def launch(job_dir: str) -> int:
    """Start the runner as a process of its own that outlives the window (and the login); returns its pid."""

    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (root, env.get("PYTHONPATH")) if p)
    env.pop("QT_QPA_PLATFORM", None)
    out = open(os.path.join(job_dir, "runner.out"), "ab")
    try:
        process = subprocess.Popen([sys.executable, "-m", "temsim_ml.batch", "run", job_dir], cwd=root, env=env,
                                   stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    finally:
        out.close()
    jobs.write_json(os.path.join(job_dir, jobs.STATUS_FILE),
                    {"state": "running", "pid": process.pid, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "current": {}, "total": 0, "done": 0, "failed": 0})
    return process.pid


def request_stop(job_dir: str) -> None:
    open(os.path.join(job_dir, jobs.STOP_FILE), "w").close()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m temsim_ml.batch run")
    parser.add_argument("job_dir")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    outcome = run_job(os.path.abspath(args.job_dir), args.device)
    return 0 if outcome in ("finished", "stopped") else 1
