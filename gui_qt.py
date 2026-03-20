import time
from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np
import torch

from PySide6 import QtCore, QtWidgets

import matplotlib

from training_parameter import TrainMethod

matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib import patches

from train_methods_gui_builder_qt import TrainMethodEditor


@dataclass
class TunableParameters:
    scale_Gr: List[float]
    rotation_Gr: float
    shift_Gr: List[float]

    scale_WS2: List[float]
    rotation_WS2: float
    shift_WS2: List[float]




@dataclass
class VizPayload:
    step: int
    loss: float
    losses: np.ndarray

    sim_img: Optional[np.ndarray] = None
    diff_img: Optional[np.ndarray] = None
    target_img: Optional[np.ndarray] = None

    ctf_phase: Optional[np.ndarray] = None
    ring_mask: Optional[np.ndarray] = None
    p_diff: Optional[np.ndarray] = None

    pot_positions: Optional[List[np.ndarray]] = None
    pot_intensities: Optional[List[np.ndarray]] = None

    hist_counts: Optional[List[np.ndarray]] = None
    hist_edges: Optional[np.ndarray] = None

    train_method: TrainMethod = None

    strain_maps: Optional[Dict[str, np.ndarray]] = None

    status: str = "idle"


class TrainingWorker(QtCore.QObject):
    progress_light = QtCore.Signal(int, float, object)
    progress_heavy = QtCore.Signal(object)
    status_changed = QtCore.Signal(str)
    finished = QtCore.Signal()

    def __init__(
        self,
        model,
        optimizer,
        target,
        brightness,
        contrast,
        pot_layer,
        ctf_layer,
        train_method,
        plot_every=5,
        heavy_every=10,
        max_steps=2_000_000,
        parent=None,
    ):
        super().__init__(parent)

        self.model = model
        self.optimizer = optimizer
        self.target = target
        self.brightness = brightness
        self.contrast = contrast
        self.pot_layer = pot_layer
        self.ctf_layer = ctf_layer
        self.train_method = train_method

        self.plot_every = plot_every
        self.heavy_every = heavy_every
        self.max_steps = max_steps

        self.show_positions_on_target = True

        self._stop = False
        self._pause = False
        self.losses = []

    def _set_optimizer_lr(self):
        lr = None
        if hasattr(self.train_method, "learning_rate"):
            lr = float(self.train_method.learning_rate)
        elif hasattr(self.train_method, "learing_rate"):
            lr = float(self.train_method.learing_rate)

        if lr is None:
            return

        if hasattr(self.optimizer, "param_groups"):
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
        elif hasattr(self.optimizer, "learning_rate"):
            self.optimizer.learning_rate = lr

    spin_values_changed = QtCore.Signal(float, float, float, float, float, float, float,n  bool)  # scale, rot, shift_x, shift_y




    @QtCore.Slot()
    def request_heavy_refresh(self):
        try:
            with torch.no_grad():
                simulated_image, target_image, p_diff, loss = self.model.loss_multislice(
                    self.target, self.brightness, self.contrast
                )

                loss_item = float(loss.detach().cpu().item())
                payload = self.build_heavy_payload(
                    step=len(self.losses) - 1 if len(self.losses) > 0 else 0,
                    loss_item=loss_item,
                    simulated_image=simulated_image,
                    target_image=target_image,
                    p_diff=p_diff,
                )
                self.progress_heavy.emit(payload)
        except Exception as exc:
            print("request_heavy_refresh failed:", exc)

    def _current_transform_group(self) -> int:
        if hasattr(self.train_method, "only_train_small_area"):
            if self.train_method.only_train_small_area == 0:
                return 0
            if self.train_method.only_train_small_area == 1:
                return 1
        return None

    def emit_current_spin_values(self):
        idx = self._current_transform_group()

        if idx is None:
            self.spin_values_changed.emit(0,0,0,0,0, False)
        else:

            with torch.no_grad():
                scale_x_val = float(self.pot_layer.scales_[idx][0].detach().cpu().item())
                scale_y_val = float(self.pot_layer.scales_[idx][1].detach().cpu().item())
                rot_val = float(self.pot_layer.rotations_[idx].detach().cpu().item())
                shift_x_val = float(self.pot_layer.shifts_[idx][0].detach().cpu().item())
                shift_y_val = float(self.pot_layer.shifts_[idx][1].detach().cpu().item())
                ux_val = float(self.pot_layer.ws2_displacement.ux.detach().cpu().numpy())
                uy_val = float(self.pot_layer.ws2_displacement.uy.detach().cpu().numpy())

            self.spin_values_changed.emit(scale_x_val, scale_y_val, rot_val, shift_x_val, shift_y_val, ux_val, uy_val, True)

    def build_heavy_payload(self, step: int, loss_item: float, simulated_image, target_image, p_diff):
        sim_img = simulated_image.detach().cpu().numpy()
        tgt_img = target_image.detach().cpu().numpy()
        diff_img = sim_img - tgt_img

        pot_positions = []
        pot_intensities = []
        for i in range(4):
            (px, py), ints = self.pot_layer.get_pos_and_ints(i)
            pos = np.column_stack([px, py]).astype(np.float32)
            ints = np.asarray(ints, dtype=np.float32)
            pot_positions.append(pos)
            pot_intensities.append(ints)

        hist_counts = None
        hist_edges = None
        intensities = self.pot_layer.get_intensities_raw()
        if intensities is not None:
            hist_counts = []
            for arr in intensities[:4]:
                arr_np = arr.detach().cpu().numpy().ravel()
                arr_np = arr_np[np.isfinite(arr_np)]
                counts, edges = np.histogram(arr_np, bins=100)
                hist_counts.append(counts.astype(np.float32))
                hist_edges = edges.astype(np.float32)

        ctf_phase = None
        ring = None
        try:
            dp, env, ring_mask, psf = self.ctf_layer.to_diffraction_patterns(
                max_angle_mrad=42.0, gpts=1300
            )
            ctf_map = dp.detach().cpu().numpy()
            ring = ring_mask.detach().cpu().numpy()
            ctf_phase = np.angle(ctf_map)
        except Exception as exc:
            print("CTF update failed:", exc)

        p_diff_np = None
        if p_diff is not None:
            p_diff_np = p_diff.detach().cpu().numpy()

        strain_maps = self.pot_layer.ws2_displacement.strain_maps()

        payload = VizPayload(
            step=step,
            loss=loss_item,
            losses=np.asarray(self.losses, dtype=np.float32).copy(),
            sim_img=sim_img,
            diff_img=diff_img,
            target_img=tgt_img,
            ctf_phase=ctf_phase,
            ring_mask=ring,
            p_diff=p_diff_np,
            pot_positions=pot_positions,
            pot_intensities=pot_intensities,
            hist_counts=hist_counts,
            hist_edges=hist_edges,
            status="running",
            train_method=self.train_method,
            strain_maps=strain_maps
        )
        return payload

    @QtCore.Slot()
    def run(self):
        self._stop = False
        self._pause = False
        self.status_changed.emit("running")

        for step in range(self.max_steps):
            if self._stop:
                break

            while self._pause and not self._stop:
                QtCore.QThread.msleep(20)

            if self._stop:
                break

            self.optimizer.zero_grad(set_to_none=True)
            self._set_optimizer_lr()

            simulated_image, target_image, p_diff, loss = self.model.loss_multislice(
                self.target, self.brightness, self.contrast
            )

            loss.backward()

            if hasattr(self.pot_layer, "zero_grads_for_frozen"):
                self.pot_layer.zero_grads_for_frozen()

            self.optimizer.step()

            loss_item = float(loss.detach().cpu().item())
            self.losses.append(loss_item)

            if step % self.plot_every == 0 or step == 0:
                self.progress_light.emit(
                    step,
                    loss_item,
                    np.asarray(self.losses, dtype=np.float32).copy()
                )
                self.emit_current_spin_values()

            if step % self.heavy_every == 0 or step == 0:
                payload = self.build_heavy_payload(
                    step=step,
                    loss_item=loss_item,
                    simulated_image=simulated_image,
                    target_image=target_image,
                    p_diff=p_diff,
                )
                self.progress_heavy.emit(payload)

            QtCore.QThread.msleep(1)

        self.status_changed.emit("stopped" if self._stop else "done")
        self.finished.emit()

    @QtCore.Slot()
    def pause(self):
        self._pause = True
        self.status_changed.emit("paused")

    @QtCore.Slot()
    def resume(self):
        self._pause = False
        self.status_changed.emit("running")

    @QtCore.Slot()
    def stop(self):
        self._stop = True
        self._pause = False
        self.status_changed.emit("stopping...")


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, target_tensor=None, parent=None):
        self.fig = Figure(figsize=(12, 8), dpi=100, constrained_layout=True)
        self.axes = [self.fig.add_subplot(3, 5, i + 1) for i in range(15)]
        super().__init__(self.fig)
        self.setParent(parent)

        for ax in self.axes:
            ax.set_xticks([])
            ax.set_yticks([])


        self.axes_cmap = {"loss": self.axes[0], "sim": self.axes[1], "diff": self.axes[2], "target": self.axes[3], "ctf": self.axes[4],
                         "pot0": self.axes[5], "pot1": self.axes[6], "pot2": self.axes[7], "pot3": self.axes[8],
                         "hist": self.axes[9], "ux": self.axes[10], "uy": self.axes[11],
                         "eps_xx": self.axes[12], "eps_yy": self.axes[13], "eps_xy": self.axes[14]}

        self.axes_cmap["loss"].set_title("Loss")
        self.axes_cmap["sim"].set_title("Simulated")
        self.axes_cmap["diff"].set_title("Diff")
        self.axes_cmap["target"].set_title("Target")
        self.axes_cmap["ctf"].set_title("CTF")
        self.axes_cmap["pot0"].set_title("Potential Gr")
        self.axes_cmap["pot1"].set_title("Potential S low")
        self.axes_cmap["pot2"].set_title("Potential W")
        self.axes_cmap["pot3"].set_title("Potential S high")
        self.axes_cmap["eps_xx"].set_title("eps xx")
        self.axes_cmap["eps_yy"].set_title("eps yy")
        self.axes_cmap["eps_xy"].set_title("eps xy")
        self.axes_cmap["ux"].set_title("ux")
        self.axes_cmap["uy"].set_title("uy")






        if target_tensor is not None:
            target_np = target_tensor.detach().cpu().numpy()
            dummy = np.zeros_like(target_np, dtype=np.float32)
        else:
            dummy = np.zeros((64, 64), dtype=np.float32)

        self.loss_line, = self.axes[0].plot([], [])
        self.axes_cmap["loss"].set_xlabel("Step")
        self.axes_cmap["loss"].set_ylabel("Loss")

        self.im_sim = self.axes_cmap["sim"].imshow(dummy, cmap="viridis", origin="lower")
        self.im_diff = self.axes_cmap["diff"].imshow(dummy, cmap="viridis", origin="lower")
        self.im_tgt = self.axes_cmap["target"].imshow(dummy, cmap="viridis", origin="lower")
        self.im_tgt_sc = self.axes_cmap["target"].scatter([], [], c="red", s=2)

        self.im_eps_xx = self.axes_cmap["eps_xx"].imshow(dummy, cmap="RdBu", origin="lower")
        self.im_eps_yy = self.axes_cmap["eps_yy"].imshow(dummy, cmap="RdBu", origin="lower")
        self.im_eps_xy = self.axes_cmap["eps_xy"].imshow(dummy, cmap="RdBu", origin="lower")
        self.im_ux = self.axes_cmap["ux"].imshow(dummy, cmap="RdBu", origin="lower")
        self.im_uy = self.axes_cmap["uy"].imshow(dummy, cmap="RdBu", origin="lower")

        self.pot_scatters = []
        for i in range(4):
            sc = self.axes_cmap["pot" + str(i)].scatter([], [], s=2, c=[], cmap="viridis")
            self.pot_scatters.append(sc)

        self.im_ctf = self.axes_cmap["ctf"].imshow(
            np.zeros((10, 10), dtype=np.float32),
            cmap="RdBu",
            vmin=-np.pi,
            vmax=np.pi,
            origin="lower",
        )
        self.im_ring = self.axes_cmap["ctf"].imshow(
            np.zeros((10, 10), dtype=np.float32),
            cmap="Greens",
            alpha=0.35,
            origin="lower",
        )

        self.hist_lines = []
        labels = ["Graphen", "S low", "W", "S high"]
        for label in labels:
            line, = self.axes_cmap["hist"].plot([], [], label=label)
            self.hist_lines.append(line)
        self.axes_cmap["hist"].legend()


        self.show_positions_on_target = True

    def set_target_rectangles(self, rectangles):
        if rectangles is None:
            return
        for x0, y0, x1, y1 in rectangles:
            rect = patches.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                linewidth=1,
                edgecolor="red",
                facecolor="none",
            )
            self.axes[3].add_patch(rect)

    def _safe_set_image(self, artist, arr):
        artist.set_data(arr)
        vmin = float(np.nanmin(arr))
        vmax = float(np.nanmax(arr))
        if not np.isfinite(vmin):
            vmin = 0.0
        if not np.isfinite(vmax):
            vmax = 1.0
        if vmin == vmax:
            vmax = vmin + 1e-12
        artist.set_clim(vmin, vmax)

    def update_light(self, step: int, loss: float, losses: np.ndarray):
        if len(losses) > 1:
            xs = np.arange(len(losses), dtype=np.int32)
            self.loss_line.set_data(xs, losses)
            self.axes[0].relim()
            self.axes[0].autoscale_view()

        self.axes[1].set_title(f"Simulated (Step {step})")
        self.axes[2].set_title(f"Diff (Step {step})")
        self.axes[3].set_title(f"Target (Step {step})")

    def get_only_area_pos(self):
        idx = self.train_method.only_train_small_area
        xmin, ymin, xmax, ymax = self.train_method.train_small_area_squares[idx]
        if idx == 0:
            pos, ints = self.pot_layer.get_pos_and_ints(0)
        else:
            pos, ints = self.pot_layer.get_pos_and_ints(1)


    def get_pos_and_ints(self, payload):

        idx = payload.train_method.only_train_small_area

        if not self.show_positions_on_target:
            return ([], []), []


        _idx = idx if idx >= 0 else 1
        if idx >= 0:
            x_min, y_min, x_max, y_max = payload.train_method.train_small_area_squares[_idx]
        else:
            x_min = 0
            y_min = 0
            x_max = payload.target_img.shape[1]
            y_max = payload.target_img.shape[0]


        if _idx == 0:
            pos = payload.pot_positions[_idx]
            ints = payload.pot_intensities[_idx]
        elif _idx == 1:
            poses = []
            intss = []
            for i in range(1,4):
                pos = payload.pot_positions[i]
                ints = payload.pot_intensities[i]
                poses.append(pos)
                intss.append(ints)
            pos = np.concatenate(poses, axis=0)
            ints = np.concatenate(intss, axis=0)

        mask = ((pos[:, 0] >= x_min) & (pos[:, 0] <= x_max) &
                (pos[:, 1] >= y_min) & (pos[:, 1] <= y_max))
        positions = pos[mask]
        intensities = ints[mask]

        return (positions[:,0] - x_min, positions[:,1]-y_min), intensities

    def update_heavy(self, payload: VizPayload, ctf_layer=None):
        if payload.sim_img is not None:
            self._safe_set_image(self.im_sim, payload.sim_img)

        if payload.diff_img is not None:
            self._safe_set_image(self.im_diff, payload.diff_img)

        if payload.target_img is not None:
            #self._safe_set_image(self.im_tgt, payload.target_img)
            self.axes_cmap["target"].clear()
            self.axes_cmap["target"].imshow(payload.target_img, cmap="viridis", origin="lower")
            self.axes_cmap["target"].set_title("Target")
            pos, ints = self.get_pos_and_ints(payload)

            idx = payload.train_method.only_train_small_area
            if idx < 0:
                ss = 1
            else:
                ss = 5

            self.axes_cmap["target"].scatter(pos[1], pos[0], c=ints, cmap="Wistia", s=ss)
            #self.im_tgt_sc.set_offsets(pos)
            #self.im_tgt_sc.set_array(ints)


        if payload.strain_maps["ux"] is not None:
            self._safe_set_image(self.im_ux, payload.strain_maps["ux"])

        if payload.strain_maps["uy"] is not None:
            self._safe_set_image(self.im_uy, payload.strain_maps["uy"])

        if payload.strain_maps["eps_xx"] is not None:
            self._safe_set_image(self.im_eps_xx, payload.strain_maps["eps_xx"])

        if payload.strain_maps["eps_yy"] is not None:
            self._safe_set_image(self.im_eps_yy, payload.strain_maps["eps_yy"])

        if payload.strain_maps["eps_xy"] is not None:
            self._safe_set_image(self.im_eps_xy, payload.strain_maps["eps_xy"])

        if payload.ctf_phase is not None:
            self.im_ctf.set_data(payload.ctf_phase)

        if payload.ring_mask is not None:
            self.im_ring.set_data(payload.ring_mask)

        if payload.pot_positions is not None and payload.pot_intensities is not None:
            for i, sc in enumerate(self.pot_scatters):
                pos = payload.pot_positions[i]
                ints = payload.pot_intensities[i]

                if pos.size == 0:
                    sc.set_offsets(np.empty((0, 2), dtype=np.float32))
                    sc.set_array(np.array([], dtype=np.float32))
                    continue

                valid_mask = np.isfinite(ints)
                pos = pos[valid_mask]
                ints = ints[valid_mask]

                if len(pos) == 0:
                    sc.set_offsets(np.empty((0, 2), dtype=np.float32))
                    sc.set_array(np.array([], dtype=np.float32))
                    continue

                sc.set_offsets(pos)
                sc.set_array(ints)

                imin = float(np.nanmin(ints))
                imax = float(np.nanmax(ints))
                if imin == imax:
                    imax = imin + 1e-12
                sc.set_clim(imin, imax)

                xmin, ymin = np.min(pos[:, 0]), np.min(pos[:, 1])
                xmax, ymax = np.max(pos[:, 0]), np.max(pos[:, 1])

                if xmin == xmax:
                    xmax = xmin + 1e-6
                if ymin == ymax:
                    ymax = ymin + 1e-6

                self.axes_cmap["pot" + str(i)].set_xlim(xmin, xmax)
                self.axes_cmap["pot" + str(i)].set_ylim(ymin, ymax)

        if payload.hist_counts is not None and payload.hist_edges is not None:
            centers = 0.5 * (payload.hist_edges[:-1] + payload.hist_edges[1:])
            ymax = 1.0
            for i, counts in enumerate(payload.hist_counts):
                self.hist_lines[i].set_data(centers, counts)
                if len(counts) > 0:
                    ymax = max(ymax, float(np.max(counts)))

            self.axes_cmap["hist"].set_xlim(float(centers[0]), float(centers[-1]))
            self.axes_cmap["hist"].set_ylim(0, ymax * 1.05)

        c10 = None
        c30 = None
        if ctf_layer is not None:
            try:
                c10 = float(ctf_layer.aberr_params["C10"].detach().cpu().numpy())
            except Exception:
                pass
            try:
                c30 = float(ctf_layer.aberr_params["C30"].detach().cpu().numpy())
            except Exception:
                pass

        if c10 is not None or c30 is not None:
            self.axes_cmap["ctf"].set_title(
                f"CTF (Step {payload.step})\n"
                f"C10: {c10 if c10 is not None else np.nan:.2f} nm   "
                f"C30: {c30 if c30 is not None else np.nan:.2f} um"
            )
        else:
            self.axes_cmap["ctf"].set_title(f"CTF (Step {payload.step})")


class LiveTrainWindow(QtWidgets.QMainWindow):
    request_heavy_refresh_signal = QtCore.Signal()
    show_positions_signal = QtCore.Signal(bool)


    def __init__(
        self,
        model,
        optimizer,
        target,
        brightness,
        contrast,
        pot_layer,
        ctf_layer,
        train_method,
        plot_every=5,
        heavy_every=10,
        max_steps=2_000_000,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Live Training Qt")

        self.model = model
        self.optimizer = optimizer
        self.target = target
        self.brightness = brightness
        self.contrast = contrast
        self.pot_layer = pot_layer
        self.ctf_layer = ctf_layer
        self.train_method = train_method

        self.plot_every = plot_every
        self.heavy_every = heavy_every
        self.max_steps = max_steps

        self.thread = None
        self.worker = None

        self._build_ui()
        self._connect_actions()
        self._apply_trainable_selection()

        if hasattr(self.train_method, "train_small_area_squares"):
            self.canvas.set_target_rectangles(self.train_method.train_small_area_squares)




    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        layout = QtWidgets.QHBoxLayout(central)

        left = QtWidgets.QVBoxLayout()
        right = QtWidgets.QVBoxLayout()

        layout.addLayout(left, stretch=4)
        layout.addLayout(right, stretch=1)

        top_tune = QtWidgets.QGroupBox("Tune")
        tune_layout = QtWidgets.QGridLayout(top_tune)

        self.scale_spin_x = QtWidgets.QDoubleSpinBox()
        self.scale_spin_x.setDecimals(5)
        self.scale_spin_x.setRange(0.80, 1.20)
        self.scale_spin_x.setSingleStep(0.01)
        self.scale_spin_x.setValue(1.12)

        self.scale_spin_y = QtWidgets.QDoubleSpinBox()
        self.scale_spin_y.setDecimals(5)
        self.scale_spin_y.setRange(0.80, 1.20)
        self.scale_spin_y.setSingleStep(0.01)
        self.scale_spin_y.setValue(1.14)

        self.shift_spin_x = QtWidgets.QDoubleSpinBox()
        self.shift_spin_x.setDecimals(5)
        self.shift_spin_x.setRange(-50.0, 50.0)
        self.shift_spin_x.setSingleStep(0.1)
        self.shift_spin_x.setValue(-9.5)

        self.shift_spin_y = QtWidgets.QDoubleSpinBox()
        self.shift_spin_y.setDecimals(5)
        self.shift_spin_y.setRange(-50.0, 50.0)
        self.shift_spin_y.setSingleStep(0.1)
        self.shift_spin_y.setValue(-11.0)

        self.rotation_spin = QtWidgets.QDoubleSpinBox()
        self.rotation_spin.setDecimals(5)
        self.rotation_spin.setRange(-np.pi/6, np.pi/6)
        self.rotation_spin.setSingleStep(0.01)
        self.rotation_spin.setValue(0.025)

        self.show_positions_checkbox = QtWidgets.QCheckBox("Show positions on target")
        self.show_positions_checkbox.setChecked(True)


        tune_layout.addWidget(QtWidgets.QLabel("scale"), 0, 0)
        tune_layout.addWidget(self.scale_spin_x, 1, 0)
        tune_layout.addWidget(self.scale_spin_y, 2, 0)

        tune_layout.addWidget(QtWidgets.QLabel("shift"), 0, 1)
        tune_layout.addWidget(self.shift_spin_x, 1, 1)
        tune_layout.addWidget(self.shift_spin_y, 2, 1)

        tune_layout.addWidget(QtWidgets.QLabel("rotation"), 0, 2)
        tune_layout.addWidget(self.rotation_spin, 1, 2)

        tune_layout.addWidget(self.show_positions_checkbox, 2, 2)



        left.addWidget(top_tune, stretch=0)

        self.canvas = MplCanvas(target_tensor=self.target)
        left.addWidget(self.canvas, stretch=1)

        status_box = QtWidgets.QGroupBox("Status")
        status_layout = QtWidgets.QFormLayout(status_box)

        self.status_label = QtWidgets.QLabel("idle")
        self.step_label = QtWidgets.QLabel("step: -")
        self.loss_label = QtWidgets.QLabel("loss: -")

        status_layout.addRow("State", self.status_label)
        status_layout.addRow("Step", self.step_label)
        status_layout.addRow("Loss", self.loss_label)

        right.addWidget(status_box)

        control_box = QtWidgets.QGroupBox("Training control")
        control_layout = QtWidgets.QVBoxLayout(control_box)

        self.start_btn = QtWidgets.QPushButton("Start")
        self.pause_btn = QtWidgets.QPushButton("Pause")
        self.resume_btn = QtWidgets.QPushButton("Resume")
        self.stop_btn = QtWidgets.QPushButton("Stop")

        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.pause_btn)
        control_layout.addWidget(self.resume_btn)
        control_layout.addWidget(self.stop_btn)

        right.addWidget(control_box)

        quick_box = QtWidgets.QGroupBox("Quick presets")
        quick_layout = QtWidgets.QHBoxLayout(quick_box)

        self.graphen_btn = QtWidgets.QPushButton("Graphen area")
        self.ws2_btn = QtWidgets.QPushButton("WS2 area")
        self.main_btn = QtWidgets.QPushButton("Main training")

        quick_layout.addWidget(self.graphen_btn)
        quick_layout.addWidget(self.ws2_btn)
        quick_layout.addWidget(self.main_btn)

        right.addWidget(quick_box)

        self.editor = TrainMethodEditor(self.train_method)
        right.addWidget(self.editor, stretch=1)

        right.addStretch(1)

        self.editor.graphen_training()
        self._apply_trainable_selection()

        self._on_scale_x_changed(self.scale_spin_x.value())
        self._on_scale_y_changed(self.scale_spin_y.value())
        self._on_shift_x_changed(self.shift_spin_x.value())
        self._on_shift_y_changed(self.shift_spin_y.value())
        self._on_rotation_changed(self.rotation_spin.value())



    def _update_spins_from_current_params(self):
        idx = self._current_transform_group()

        if idx is not None:

            with torch.no_grad():
                scale_x_val = float(self.pot_layer.scales_[idx][0].detach().cpu().item())
                scale_y_val = float(self.pot_layer.scales_[idx][1].detach().cpu().item())
                rot_val = float(self.pot_layer.rotations_[idx].detach().cpu().item())
                shift_x_val = float(self.pot_layer.shifts_[idx][0].detach().cpu().item())
                shift_y_val = float(self.pot_layer.shifts_[idx][1].detach().cpu().item())

            widgets_and_values = [
                (self.scale_spin_x, scale_x_val),
                (self.scale_spin_y, scale_y_val),
                (self.rotation_spin, rot_val),
                (self.shift_spin_x, shift_x_val),
                (self.shift_spin_y, shift_y_val),
            ]

            for widget, value in widgets_and_values:
                widget.blockSignals(True)
                widget.setValue(value)
                widget.blockSignals(False)


    def _current_transform_group(self):
        if self.train_method.only_train_small_area >= 0:
            return self.train_method.only_train_small_area
        else:
            None

    def _on_show_positions_changed(self, checked: bool):
        #self.show_positions_checkbox.setChecked(checked)
        self.canvas.show_positions_on_target = checked
        self._trigger_heavy_refresh()

    def _on_scale_x_changed(self, value: float):
        with torch.no_grad():
            # 0 = Graphen, 1 = WS2
            idx = self._current_transform_group()
            if idx is not None:
                self.pot_layer.scales_[idx][0] = float(value)

        self._trigger_heavy_refresh()

    def _on_scale_y_changed(self, value: float):
        with torch.no_grad():
            idx = self._current_transform_group()
            if idx is not None:
                self.pot_layer.scales_[idx][1] = float(value)

        self._trigger_heavy_refresh()

    def _on_rotation_changed(self, value: float):
        with torch.no_grad():
            idx = self._current_transform_group()
            if idx is not None:
                self.pot_layer.rotations_[idx][0] = float(value)

        self._trigger_heavy_refresh()

    def _on_shift_x_changed(self, value: float):
        with torch.no_grad():
            idx = self._current_transform_group()
            if idx is not None:
                # beide Komponenten gleich setzen
                self.pot_layer.shifts_[idx][0] = float(value)

        self._trigger_heavy_refresh()


    def _on_shift_y_changed(self, value: float):
        with torch.no_grad():
            idx = self._current_transform_group()
            if idx is not None:
                # beide Komponenten gleich setzen
                self.pot_layer.shifts_[idx][1] = float(value)

        self._trigger_heavy_refresh()

    @QtCore.Slot(float, float, float, float)
    def on_spin_values_changed(self, scale_x_val, scale_y_val, rot_val, shift_x_val, shift_y_val, activated):
        self._update_spins_from_values(scale_x_val, scale_y_val, rot_val, shift_x_val, shift_y_val, activated)

    def _set_spin_value_silent(self, spinbox, value: float):
        spinbox.blockSignals(True)
        spinbox.setValue(value)
        spinbox.blockSignals(False)

    def _update_spins_from_values(self, scale_x_val, scale_y_val, rot_val, shift_x_val, shift_y_val, activated):
        self._set_spin_value_silent(self.scale_spin_x, scale_x_val)
        self._set_spin_value_silent(self.scale_spin_y, scale_y_val)
        self._set_spin_value_silent(self.rotation_spin, rot_val)
        self._set_spin_value_silent(self.shift_spin_x, shift_x_val)
        self._set_spin_value_silent(self.shift_spin_y, shift_y_val)

        self.scale_spin_x.setEnabled(activated)
        self.scale_spin_y.setEnabled(activated)
        self.rotation_spin.setEnabled(activated)
        self.shift_spin_x.setEnabled(activated)
        self.shift_spin_y.setEnabled(activated)

    def _connect_actions(self):
        self.start_btn.clicked.connect(self.start_training)
        self.pause_btn.clicked.connect(self.pause_training)
        self.resume_btn.clicked.connect(self.resume_training)
        self.stop_btn.clicked.connect(self.stop_training)

        self.graphen_btn.clicked.connect(self._graphen_training)
        self.ws2_btn.clicked.connect(self._ws2_training)
        self.main_btn.clicked.connect(self._main_training)

        self.editor.changed.connect(self._apply_trainable_selection)

        self.scale_spin_x.valueChanged.connect(self._on_scale_x_changed)
        self.scale_spin_y.valueChanged.connect(self._on_scale_y_changed)
        self.shift_spin_x.valueChanged.connect(self._on_shift_x_changed)
        self.shift_spin_y.valueChanged.connect(self._on_shift_y_changed)
        self.rotation_spin.valueChanged.connect(self._on_rotation_changed)
        self.show_positions_checkbox.stateChanged.connect(self._on_show_positions_changed)

    def _apply_trainable_selection(self):
        if hasattr(self.pot_layer, "apply_trainable_selection"):
            self.pot_layer.apply_trainable_selection()

    def _graphen_training(self):
        self.editor.graphen_training()
        self._apply_trainable_selection()
        self._update_spins_from_current_params()

    def _ws2_training(self):
        self.editor.WS2_training()
        self._apply_trainable_selection()
        self._update_spins_from_current_params()

    def _main_training(self):
        self.editor.main_training()
        self._apply_trainable_selection()
        self._update_spins_from_current_params()

    def start_training(self):
        if self.thread is not None and self.thread.isRunning():
            return

        self.thread = QtCore.QThread(self)
        self.worker = TrainingWorker(
            model=self.model,
            optimizer=self.optimizer,
            target=self.target,
            brightness=self.brightness,
            contrast=self.contrast,
            pot_layer=self.pot_layer,
            ctf_layer=self.ctf_layer,
            train_method=self.train_method,
            plot_every=self.plot_every,
            heavy_every=self.heavy_every,
            max_steps=self.max_steps,
        )
        self.worker.moveToThread(self.thread)

        self.worker.spin_values_changed.connect(self.on_spin_values_changed)

        self.request_heavy_refresh_signal.connect(self.worker.request_heavy_refresh)

        self.thread.started.connect(self.worker.run)
        self.worker.progress_light.connect(self.on_progress_light)
        self.worker.progress_heavy.connect(self.on_progress_heavy)
        self.worker.status_changed.connect(self.on_status_changed)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.finished.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)

        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.resume_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.thread.start()

    def _do_local_heavy_refresh(self):
        try:
            with torch.no_grad():
                simulated_image, target_image, p_diff, loss = self.model.loss_multislice(
                    self.target, self.brightness, self.contrast
                )
                loss_item = float(loss.detach().cpu().item())

                payload = self._build_local_heavy_payload(
                    step=0,
                    loss_item=loss_item,
                    simulated_image=simulated_image,
                    target_image=target_image,
                    p_diff=p_diff
                )
                self.on_progress_heavy(payload)
        except Exception as exc:
            print("local heavy refresh failed:", exc)

    def _build_local_heavy_payload(self, step, loss_item, simulated_image, target_image, p_diff):
        sim_img = simulated_image.detach().cpu().numpy()
        tgt_img = target_image.detach().cpu().numpy()
        diff_img = sim_img - tgt_img

        pot_positions = []
        pot_intensities = []
        for i in range(4):
            (px, py), ints = self.pot_layer.get_pos_and_ints(i)
            pos = np.column_stack([px, py]).astype(np.float32)
            ints = np.asarray(ints, dtype=np.float32)
            pot_positions.append(pos)
            pot_intensities.append(ints)

        hist_counts = []
        hist_edges = None
        intensities = self.pot_layer.get_intensities_raw()
        if intensities is not None:
            for arr in intensities[:4]:
                arr_np = arr.detach().cpu().numpy().ravel()
                arr_np = arr_np[np.isfinite(arr_np)]
                counts, edges = np.histogram(arr_np, bins=100)
                hist_counts.append(counts.astype(np.float32))
                hist_edges = edges.astype(np.float32)
        else:
            hist_counts = None
            hist_edges = None

        ctf_phase = None
        ring = None
        try:
            dp, env, ring_mask, psf = self.ctf_layer.to_diffraction_patterns(
                max_angle_mrad=42.0, gpts=1300
            )
            ctf_map = dp.detach().cpu().numpy()
            ring = ring_mask.detach().cpu().numpy()
            ctf_phase = np.angle(ctf_map)
        except Exception as exc:
            print("local CTF update failed:", exc)

        p_diff_np = None if p_diff is None else p_diff.detach().cpu().numpy()

        strain_maps = self.pot_layer.ws2_displacement.strain_maps()

        return VizPayload(
            step=step,
            loss=loss_item,
            losses=np.array([], dtype=np.float32),
            sim_img=sim_img,
            diff_img=diff_img,
            target_img=tgt_img,
            ctf_phase=ctf_phase,
            ring_mask=ring,
            p_diff=p_diff_np,
            pot_positions=pot_positions,
            pot_intensities=pot_intensities,
            hist_counts=hist_counts,
            hist_edges=hist_edges,
            status="preview",
            train_method=self.train_method,
            strain_maps=strain_maps
        )

    def _trigger_heavy_refresh(self):
        if self.worker is not None:
            self.request_heavy_refresh_signal.emit()
        else:
            self._do_local_heavy_refresh()

    def pause_training(self):
        if self.worker is not None:
            self.worker.pause()
            self.pause_btn.setEnabled(False)
            self.resume_btn.setEnabled(True)

    def resume_training(self):
        if self.worker is not None:
            self.worker.resume()
            self.pause_btn.setEnabled(True)
            self.resume_btn.setEnabled(False)

    def stop_training(self):
        if self.worker is not None:
            self.worker.stop()
            self.status_label.setText("stopping...")
            self.pause_btn.setEnabled(False)
            self.resume_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)

    @QtCore.Slot(int, float, object)
    def on_progress_light(self, step, loss, losses):
        self.step_label.setText(f"step: {step}")
        self.loss_label.setText(f"loss: {loss:.6g}")
        self.canvas.update_light(step, loss, losses)
        self.canvas.draw()
        self.canvas.flush_events()

    @QtCore.Slot(object)
    def on_progress_heavy(self, payload: VizPayload):
        self.step_label.setText(f"step: {payload.step}")
        self.loss_label.setText(f"loss: {payload.loss:.6g}")
        self.canvas.update_heavy(payload, ctf_layer=self.ctf_layer)
        self.canvas.draw()
        self.canvas.flush_events()

    @QtCore.Slot(str)
    def on_status_changed(self, status):
        self.status_label.setText(status)

    @QtCore.Slot()
    def on_worker_finished(self):
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None

        self.thread = None


def launch_gui_qt(
    model,
    optimizer,
    target,
    brightness,
    contrast,
    pot_layer,
    ctf_layer,
    train_method,
    plot_every=5,
    heavy_every=10,
    max_steps=2_000_000,
):
    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QtWidgets.QApplication([])

    window = LiveTrainWindow(
        model=model,
        optimizer=optimizer,
        target=target,
        brightness=brightness,
        contrast=contrast,
        pot_layer=pot_layer,
        ctf_layer=ctf_layer,
        train_method=train_method,
        plot_every=plot_every,
        heavy_every=heavy_every,
        max_steps=max_steps,
    )
    window.resize(1600, 950)
    window.show()

    if owns_app:
        app.exec()

    return window