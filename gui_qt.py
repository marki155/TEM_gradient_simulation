import time
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
from scipy.ndimage import gaussian_filter

import torch

from PySide6 import QtCore, QtWidgets

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib import patches

from train_methods_gui_builder_qt import TrainMethodEditor


@dataclass
class VizPayload:
    step: int
    loss: float
    losses: Optional[np.ndarray] = None

    sim_img: Optional[np.ndarray] = None
    diff_img: Optional[np.ndarray] = None
    target_img: Optional[np.ndarray] = None

    ctf_phase: Optional[np.ndarray] = None
    ring_mask: Optional[np.ndarray] = None
    p_diff: Optional[np.ndarray] = None

    pot_positions: Optional[List[np.ndarray]] = None   # [N,2] je Klasse
    pot_intensities: Optional[List[np.ndarray]] = None # [N] je Klasse

    hist_counts: Optional[List[np.ndarray]] = None
    hist_edges: Optional[np.ndarray] = None

    status: str = "idle"


class TrainingWorker(QtCore.QObject):
    progress_light = QtCore.Signal(int, float, object)   # step, loss, losses
    progress_heavy = QtCore.Signal(object)               # VizPayload
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
        plot_every=10,
        heavy_every=50,
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

        self._stop = False
        self._pause = False
        self.losses = []

    @QtCore.Slot()
    def run(self):
        self._stop = False
        self._pause = False
        self.status_changed.emit("running")

        for step in range(self.max_steps):
            if self._stop:
                break

            while self._pause and not self._stop:
                time.sleep(0.03)

            if self._stop:
                break

            self.optimizer.zero_grad(set_to_none=True)

            if hasattr(self.train_method, "learning_rate"):
                lr = float(self.train_method.learning_rate)
            elif hasattr(self.train_method, "learing_rate"):
                lr = float(self.train_method.learing_rate)
            else:
                lr = None

            if lr is not None:
                if hasattr(self.optimizer, "param_groups"):
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = lr
                elif hasattr(self.optimizer, "learning_rate"):
                    self.optimizer.learning_rate = lr

            simulated_image, target_image, p_diff, loss = self.model.loss_multislice(
                self.target, self.brightness, self.contrast
            )

            loss.backward()

            if hasattr(self.pot_layer, "zero_grads_for_frozen"):
                self.pot_layer.zero_grads_for_frozen()

            self.optimizer.step()

            loss_item = float(loss.detach().cpu().item())
            self.losses.append(loss_item)

            if step % self.plot_every == 0:
                self.progress_light.emit(
                    step,
                    loss_item,
                    np.asarray(self.losses, dtype=np.float32).copy()
                )

            if step % self.heavy_every == 0 or step == 0:
                sim_img = simulated_image.detach().cpu().numpy()
                tgt_img = target_image.detach().cpu().numpy()
                diff_img = sim_img - tgt_img

                pot_positions = []
                pot_intensities = []
                for i in range(4):
                    pos, ints = self.pot_layer.get_pos_and_ints(i)
                    pos_np = np.column_stack([np.asarray(pos[0]), np.asarray(pos[1])]).astype(np.float32)
                    ints_np = np.asarray(ints, dtype=np.float32)
                    pot_positions.append(pos_np)
                    pot_intensities.append(ints_np)

                hist_counts = None
                hist_edges = None
                intensities = self.pot_layer.get_intensities_raw()
                if intensities is not None:
                    hist_counts = []
                    hist_edges = None
                    for arr in intensities[:4]:
                        arr_np = arr.detach().cpu().numpy().ravel()
                        counts, edges = np.histogram(arr_np, bins=100)
                        hist_counts.append(counts.astype(np.float32))
                        hist_edges = edges.astype(np.float32)

                ctf_phase = None
                ring = None
                p_diff_np = None

                try:
                    dp, env, ring_mask, psf = self.ctf_layer.to_diffraction_patterns(
                        max_angle_mrad=42.0, gpts=1300
                    )
                    ctf_map = dp.detach().cpu().numpy()
                    ring = ring_mask.detach().cpu().numpy()
                    ctf_phase = np.angle(ctf_map)
                except Exception:
                    pass

                if p_diff is not None:
                    p_diff_np = p_diff.detach().cpu().numpy()

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
                )
                self.progress_heavy.emit(payload)

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
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(12, 8), dpi=100, constrained_layout=True)
        self.axes = [self.fig.add_subplot(3, 4, i + 1) for i in range(11)]
        super().__init__(self.fig)
        self.setParent(parent)

        for ax in self.axes:
            ax.set_xticks([])
            ax.set_yticks([])

        self.axes[0].set_title("Loss")
        self.axes[1].set_title("Simulated")
        self.axes[2].set_title("Diff (MSE)")
        self.axes[3].set_title("Target")
        self.axes[4].set_title("Potential 0")
        self.axes[5].set_title("Potential 1")
        self.axes[6].set_title("Potential 2")
        self.axes[7].set_title("Potential 3")
        self.axes[8].set_title("CTF phase + ringmask")
        self.axes[9].set_title("Intensities")
        self.axes[10].set_title("Diff Power spectrum")

        dummy = np.zeros((10, 10), dtype=np.float32)

        self.loss_line, = self.axes[0].plot([], [])
        self.axes[0].set_xlabel("Step")
        self.axes[0].set_ylabel("Loss")

        self.im_sim = self.axes[1].imshow(dummy, cmap="viridis", origin="lower", animated=False)
        self.im_diff = self.axes[2].imshow(dummy, cmap="viridis", origin="lower", animated=False)
        self.im_tgt = self.axes[3].imshow(dummy, cmap="viridis", origin="lower", animated=False)

        self.pot_scatters = []
        for i in range(4):
            sc = self.axes[4 + i].scatter([], [], s=2, c=[], cmap="viridis")
            self.pot_scatters.append(sc)

        self.im_ctf = self.axes[8].imshow(dummy, cmap="RdBu", vmin=-np.pi, vmax=np.pi, origin="lower")
        self.im_ring = self.axes[8].imshow(dummy, cmap="Greens", alpha=0.35, origin="lower")

        self.hist_lines = []
        labels = ["Graphen", "S low", "W", "S high"]
        for label in labels:
            line, = self.axes[9].plot([], [], label=label)
            self.hist_lines.append(line)
        self.axes[9].legend()

        self.im_pdiff = self.axes[10].imshow(dummy, cmap="viridis", origin="lower", aspect="auto")

    def set_target_rectangles(self, rectangles):
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

    def update_light(self, step: int, loss: float, losses: np.ndarray):
        if len(losses) > 1:
            xs = np.arange(len(losses), dtype=np.int32)
            self.loss_line.set_data(xs, losses)
            self.axes[0].relim()
            self.axes[0].autoscale_view()

        self.axes[1].set_title(f"Simulated (Step {step})")
        self.axes[2].set_title(f"Diff (Step {step})")
        self.axes[3].set_title(f"Target (Step {step})")

    def update_heavy(self, payload: VizPayload, ctf_layer=None):
        if payload.sim_img is not None:
            self.im_sim.set_data(payload.sim_img)
            self.im_sim.set_clim(np.min(payload.sim_img), np.max(payload.sim_img))

        if payload.diff_img is not None:
            self.im_diff.set_data(payload.diff_img)
            self.im_diff.set_clim(np.min(payload.diff_img), np.max(payload.diff_img))

        if payload.target_img is not None:
            self.im_tgt.set_data(payload.target_img)
            self.im_tgt.set_clim(np.min(payload.target_img), np.max(payload.target_img))

        if payload.p_diff is not None:
            img = payload.p_diff[0] if payload.p_diff.ndim >= 3 else payload.p_diff
            self.im_pdiff.set_data(img)
            self.im_pdiff.set_clim(np.min(img), np.max(img))

        if payload.ctf_phase is not None:
            self.im_ctf.set_data(payload.ctf_phase)

        if payload.ring_mask is not None:
            self.im_ring.set_data(payload.ring_mask)

        if payload.pot_positions is not None and payload.pot_intensities is not None:
            for i, sc in enumerate(self.pot_scatters):
                pos = payload.pot_positions[i]
                ints = payload.pot_intensities[i]
                if pos.size == 0:
                    sc.set_offsets(np.empty((0, 2)))
                    sc.set_array(np.array([], dtype=np.float32))
                else:
                    sc.set_offsets(pos)
                    sc.set_array(ints)

                self.axes[4 + i].relim()
                self.axes[4 + i].autoscale_view()

        if payload.hist_counts is not None and payload.hist_edges is not None:
            centers = 0.5 * (payload.hist_edges[:-1] + payload.hist_edges[1:])
            ymax = 1
            for i, counts in enumerate(payload.hist_counts):
                self.hist_lines[i].set_data(centers, counts)
                if len(counts) > 0:
                    ymax = max(ymax, float(np.max(counts)))
            self.axes[9].set_xlim(float(centers[0]), float(centers[-1]))
            self.axes[9].set_ylim(0, ymax * 1.05)

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
            self.axes[8].set_title(
                f"CTF (Step {payload.step})\nC10: {c10 if c10 is not None else np.nan:.2f} nm   "
                f"C30: {c30 if c30 is not None else np.nan:.2f} um"
            )
        else:
            self.axes[8].set_title(f"CTF (Step {payload.step})")

        self.draw_idle()


class LiveTrainWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        model,
        optimizer,
        target: torch.Tensor,
        brightness,
        contrast,
        pot_layer,
        ctf_layer,
        train_method,
        plot_every=10,
        heavy_every=50,
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

        if hasattr(self.train_method, "train_small_area_squares"):
            self.canvas.set_target_rectangles(self.train_method.train_small_area_squares)

        self._apply_trainable_selection()
        self.editor.graphen_training()

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

        self.scale_spin = QtWidgets.QDoubleSpinBox()
        self.shift_spin = QtWidgets.QDoubleSpinBox()
        self.rotation_spin = QtWidgets.QDoubleSpinBox()

        for i, (label, widget) in enumerate([
            ("scale", self.scale_spin),
            ("shift", self.shift_spin),
            ("rotation", self.rotation_spin),
        ]):
            tune_layout.addWidget(QtWidgets.QLabel(label), 0, i)
            tune_layout.addWidget(widget, 1, i)

        left.addWidget(top_tune, stretch=0)

        self.canvas = MplCanvas()
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

    def _connect_actions(self):
        self.start_btn.clicked.connect(self.start_training)
        self.pause_btn.clicked.connect(self.pause_training)
        self.resume_btn.clicked.connect(self.resume_training)
        self.stop_btn.clicked.connect(self.stop_training)

        self.graphen_btn.clicked.connect(self._graphen_training)
        self.ws2_btn.clicked.connect(self._ws2_training)
        self.main_btn.clicked.connect(self._main_training)

        self.editor.changed.connect(self._apply_trainable_selection)

    def _apply_trainable_selection(self):
        if hasattr(self.pot_layer, "apply_trainable_selection"):
            self.pot_layer.apply_trainable_selection()

    def _graphen_training(self):
        self.editor.graphen_training()
        self._apply_trainable_selection()

    def _ws2_training(self):
        self.editor.WS2_training()
        self._apply_trainable_selection()

    def _main_training(self):
        self.editor.main_training()
        self._apply_trainable_selection()

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
        self.canvas.draw_idle()

    @QtCore.Slot(object)
    def on_progress_heavy(self, payload: VizPayload):
        self.step_label.setText(f"step: {payload.step}")
        self.loss_label.setText(f"loss: {payload.loss:.6g}")
        self.canvas.update_heavy(payload, ctf_layer=self.ctf_layer)

    @QtCore.Slot(str)
    def on_status_changed(self, status):
        self.status_label.setText(status)

    @QtCore.Slot()
    def on_worker_finished(self):
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

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
    plot_every=10,
    heavy_every=50,
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