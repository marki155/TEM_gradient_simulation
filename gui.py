import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Optional

import numpy as np
import tkinter as tk
from tkinter import ttk

import matplotlib
from matplotlib import patches

from train_methods_gui_builder import TrainMethodEditor

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import torch

# falls du gaussian_filter nutzt:
from scipy.ndimage import gaussian_filter


@dataclass
class SharedVizState:

    step: int = 0
    loss: float = np.nan
    losses: List[float] = None

    sim_img: Optional[np.ndarray] = None
    diff_img: Optional[np.ndarray] = None
    target_img: Optional[np.ndarray] = None

    pot_imgs: List[Optional[np.ndarray]] = None  # len 4
    ctf_phase: Optional[np.ndarray] = None
    ring_mask: Optional[np.ndarray] = None
    intensities: Optional[np.ndarray] = None
    p_diff: Optional[np.ndarray] = None

    # lock for thread-safe access
    lock: threading.Lock = threading.Lock()

    status: str = "idle"


class LiveTrainGUI:
    """
    GUI, die deinen bestehenden Trainings-Loop in einem Thread laufen lässt
    und alle 5 Steps die 9 Panels aktualisiert.
    """

    def __init__(
        self,
        root: tk.Tk,
        model,
        optimizer,
        target: torch.Tensor,
        brightness,
        contrast,
        pot_layer,
        ctf_layer,
        plot_every: int = 5,
        max_steps: int = 2_000_000,
        intensity_classes_getter: Optional[Callable[[], str]] = None,
        train_method=None,
        train_small_area_square = None
    ):
        self.root = root
        self.root.title("Live TEM Simulation Optimizer")

        self.model = model
        self.optimizer = optimizer
        self.target = target
        self.brightness = brightness
        self.contrast = contrast
        self.pot_layer = pot_layer
        self.ctf_layer = ctf_layer

        self.plot_every = plot_every
        self.max_steps = max_steps
        self.train_method = train_method

        # Optional: wenn du pot_layer.intensity_classes nutzt
        self.intensity_classes_getter = intensity_classes_getter

        # Thread control
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._trainer_thread: Optional[threading.Thread] = None
        self.train_small_area_square = train_method.train_small_area_squares

        # Shared state for visualization
        self.viz = SharedVizState(
            losses=[],
            pot_imgs=[None, None, None, None],
        )



        # GUI vars
        self.status_var = tk.StringVar(value="idle")
        self.step_var = tk.StringVar(value="step: -")
        self.loss_var = tk.StringVar(value="loss: -")

        self.group_vars: List[tk.BooleanVar] = []

        self._build_ui()
        self._apply_trainable_selection()  # initial

        # periodic refresh
        self._refresh_ms = 100
        self._schedule_refresh()



    def _apply_trainable_selection(self):
        self.pot_layer.apply_trainable_selection()


    def _zero_grads_for_frozen(self):
        self.pot_layer.zero_grads_for_frozen()



    # -------------------------
    # UI
    # -------------------------
    def _build_ui(self):
        main = ttk.Frame(self.root, padding=8)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        plot_frame = ttk.Frame(main)
        self.ctrl_frame = ttk.Frame(main)

        plot_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.ctrl_frame.grid(row=0, column=1, sticky="ns")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

        # ---- 3x3 figure
        self.fig = Figure(figsize=(9.5, 7.5), dpi=100)
        self.axes = [self.fig.add_subplot(3, 4, i + 1) for i in range(11)]

        for ax in self.axes:
            ax.set_xticks([])
            ax.set_yticks([])

        self.axes[0].set_title("Loss")
        self.axes[1].set_title("Simulated")
        self.axes[2].set_title("Diff (MSE)")
        self.axes[3].set_title("Target")
        self.axes[4].set_title("Potential 0 (blur)")
        self.axes[5].set_title("Potential 1 (blur)")
        self.axes[6].set_title("Potential 2 (blur)")
        self.axes[7].set_title("Potential 3 (blur)")
        self.axes[8].set_title("CTF phase + ringmask")
        self.axes[9].set_title("Intensities")
        self.axes[10].set_title("Diff Power spectrum (optional)")

        self.fig.tight_layout(pad=1.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Prepare image handles (for speed)
        dummy_img = np.zeros_like(self.target.detach().cpu())

        self.im_sim = self.axes[1].imshow(dummy_img, cmap="viridis", origin="lower")
        self.im_diff = self.axes[2].imshow(dummy_img, cmap="viridis", origin="lower")
        self.im_tgt = self.axes[3].imshow(self.target.detach().cpu(), cmap="viridis", origin="lower")

        for i in range(len(self.train_small_area_square)):
            x0, y0, x1, y1 = self.train_small_area_square[i]

            width = x1 - x0
            height = y1 - y0

            rect = patches.Rectangle(
                (x0, y0), width, height,
                linewidth=1,
                edgecolor='red',
                facecolor='none'
            )

            self.axes[3].add_patch(rect)

        self.im_pots = [self.axes[i].scatter([],[]) for i in range(4, 8)]
        self.im_ctf = self.axes[8].imshow(np.zeros((10, 10)), cmap="RdBu", vmin=-np.pi, vmax=np.pi)
        self.im_ring = self.axes[8].imshow(np.zeros((10, 10)), cmap="Greens", alpha=0.4, origin="lower")
        self.hist_gr = self.axes[9].hist([], bins=100, alpha=0.5, label='Graphen')
        self.hist_S1 = self.axes[9].hist([], bins=100, alpha=0.5, label='S low')
        self.hist_W = self.axes[9].hist([], bins=100, alpha=0.5, label='W')
        self.hist_S2 = self.axes[9].hist([], bins=100, alpha=0.5, label='S high')
        self.axes[9].legend()

        # Loss line
        self.loss_line, = self.axes[0].plot([], [])
        self.axes[0].set_xlabel("Step")
        self.axes[0].set_ylabel("Loss")

        # ---- controls
        status_box = ttk.LabelFrame(self.ctrl_frame, text="Status", padding=8)
        status_box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.ctrl_frame.columnconfigure(0, weight=1)

        ttk.Label(status_box, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        ttk.Label(status_box, textvariable=self.step_var).grid(row=1, column=0, sticky="w")
        ttk.Label(status_box, textvariable=self.loss_var).grid(row=2, column=0, sticky="w")

        btn_box = ttk.LabelFrame(self.ctrl_frame, text="Training control", padding=8)
        btn_box.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        btn_box.columnconfigure(0, weight=1)

        self.start_btn = ttk.Button(btn_box, text="Start", command=self.start)
        self.pause_btn = ttk.Button(btn_box, text="Pause", command=self.pause, state="disabled")
        self.resume_btn = ttk.Button(btn_box, text="Resume", command=self.resume, state="disabled")
        self.stop_btn = ttk.Button(btn_box, text="Stop", command=self.stop, state="disabled")

        self.start_btn.grid(row=0, column=0, sticky="ew")
        self.pause_btn.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.resume_btn.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        self.stop_btn.grid(row=3, column=0, sticky="ew", pady=(6, 0))



        self.train_method_editor = TrainMethodEditor(self.ctrl_frame, self.train_method,
                                                     callback=self._apply_trainable_selection)
        self.train_method_editor.frame.grid(row=4, column=0, sticky="ew", pady=(8, 0))

        quick = ttk.Frame(self.ctrl_frame)
        quick.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(quick, text="Graphen area", command=self._graphen_training).grid(row=0, column=0, sticky="ew")
        ttk.Button(quick, text="WS2 area", command=self._ws2_training).grid(row=0, column=1, sticky="ew", padx=(6, 0))
        ttk.Button(quick, text="main training", command=self._main_training).grid(row=0, column=2, sticky="ew", padx=(6, 0))
        quick.columnconfigure(0, weight=1)
        quick.columnconfigure(1, weight=1)

        self._graphen_training()

    def _on_checkbox_change(self):
        self._apply_trainable_selection()

    def _graphen_training(self):
        self.train_method_editor.graphen_training()
        self._apply_trainable_selection()

    def _ws2_training(self):
        self.train_method_editor.WS2_training()
        self._apply_trainable_selection()

    def _main_training(self):
        self.train_method_editor.main_training()
        self._apply_trainable_selection()



    # -------------------------
    # Training Thread
    # -------------------------
    def _training_loop(self):
        self._stop_event.clear()
        self._pause_event.clear()

        #self.status_var.set("running")
        with self.viz.lock:
            self.viz.status = "running"

        # Your loop adapted:
        for step in range(self.max_steps):
            if self._stop_event.is_set():
                break

            while self._pause_event.is_set() and not self._stop_event.is_set():
                time.sleep(0.05)

            # TRAIN STEP
            self.optimizer.zero_grad(set_to_none=True)
            self.optimizer.learning_rate = float(self.train_method.learning_rate)

            # === your forward/loss ===
            # simulated_image, loss = self.model.loss(self.target, self.brightness, self.contrast)
            simulated_image, target_image, P_diff, loss = self.model.loss_multislice(self.target, self.brightness, self.contrast)

            loss.backward()

            # live gating:
            self._zero_grads_for_frozen()

            self.optimizer.step()

            # optionally:
            # self.model.clamp_parameters()

            # record
            loss_item = float(loss.detach().cpu().item())
            with self.viz.lock:
                self.viz.step = step
                self.viz.loss = loss_item
                self.viz.losses.append(loss_item)

            # VIS UPDATE EVERY N STEPS
            if step % self.plot_every == 0 or step == 0:
                # --- compute sim/diff/target
                sim_img = simulated_image.clone().detach().cpu().numpy()
                tgt_img = target_image.clone().detach().cpu().numpy()

                #train_area_idx = int(self.train_method.only_train_small_area)
                #if train_area_idx >= 0 and train_area_idx < len(self.train_method.train_small_area_squares):
                #    x_min, y_min, x_max, y_max = self.train_method.train_small_area_squares[train_area_idx]
                #    target_image_selected = tgt_img[x_min:x_max, y_min:y_max]
                #else:
                #    target_image_selected = tgt_img
#
#                if self.train_method.show_selected_area_target:
#                    tgt_img = target_image_selected



                diff_img = (sim_img - tgt_img)

                # --- potentials
                potentials, _ = self.pot_layer(None)  # as in your code
                pot_imgs = []
                for pot_i in potentials[:4]:
                    pot_i_blurred = gaussian_filter(pot_i.detach().cpu().numpy(), sigma=5.0)
                    pot_imgs.append(pot_i_blurred)

                while len(pot_imgs) < 4:
                    pot_imgs.append(None)

                intensities = self.pot_layer.get_intensities_raw()

                # --- CTF
                dp, env, ring_mask, psf = self.ctf_layer.to_diffraction_patterns(max_angle_mrad=42.0, gpts=1300)
                ctf_map = dp.detach().cpu().numpy()
                ring = ring_mask.detach().cpu().numpy()

                ctf_phase = np.angle(ctf_map)

                # push to shared
                with self.viz.lock:
                    self.viz.sim_img = sim_img
                    self.viz.diff_img = diff_img
                    self.viz.target_img = tgt_img
                    self.viz.pot_imgs = pot_imgs
                    self.viz.ctf_phase = ctf_phase
                    self.viz.ring_mask = ring
                    self.viz.intensities = intensities
                    if P_diff is not None:
                        self.viz.p_diff = P_diff.detach().cpu().numpy()
                    else:
                        self.viz.p_diff = None

                if step == 0:
                    self.init_images()
            # optional small sleep to keep UI smooth if needed:
            # time.sleep(0.001)

        #self.status_var.set("stopped" if self._stop_event.is_set() else "done")
        with self.viz.lock:
            self.viz.status = "stopped" if self._stop_event.is_set() else "done"

    # -------------------------
    # Buttons
    # -------------------------
    def start(self):
        if self._trainer_thread and self._trainer_thread.is_alive():
            return

        self._stop_event.clear()
        self._pause_event.clear()

        self.start_btn.configure(state="disabled")
        self.pause_btn.configure(state="normal")
        self.resume_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

        self._trainer_thread = threading.Thread(target=self._training_loop, daemon=True)
        self._trainer_thread.start()

    def pause(self):
        self._pause_event.set()
        self.status_var.set("paused")
        self.pause_btn.configure(state="disabled")
        self.resume_btn.configure(state="normal")

    def resume(self):
        self._pause_event.clear()
        self.status_var.set("running")
        self.pause_btn.configure(state="normal")
        self.resume_btn.configure(state="disabled")

    def stop(self):
        self._stop_event.set()
        self._pause_event.clear()
        self.status_var.set("stopping...")
        self.pause_btn.configure(state="disabled")
        self.resume_btn.configure(state="disabled")
        self.stop_btn.configure(state="disabled")
        self.start_btn.configure(state="normal")

    # -------------------------
    # GUI refresh
    # -------------------------
    def _schedule_refresh(self):
        self.root.after(self._refresh_ms, self._refresh_ui)

    def _refresh_ui(self):
        # Pull shared data
        with self.viz.lock:
            step = self.viz.step
            loss = self.viz.loss
            losses = list(self.viz.losses)
            sim = self.viz.sim_img
            diff = self.viz.diff_img
            tgt = self.viz.target_img
            pots = list(self.viz.pot_imgs) if self.viz.pot_imgs is not None else [None]*4
            ctf_phase = self.viz.ctf_phase
            ring = self.viz.ring_mask
            status = self.viz.status
            intensities = self.viz.intensities
            P_diff = self.viz.p_diff


        self.status_var.set(status)
        self.step_var.set(f"step: {step}")
        self.loss_var.set(f"loss: {loss:.6g}" if np.isfinite(loss) else "loss: -")

        # Labels
        self.step_var.set(f"step: {step}")
        if np.isfinite(loss):
            self.loss_var.set(f"loss: {loss:.6g}")
        else:
            self.loss_var.set("loss: -")

        # Loss plot
        if len(losses) > 1:
            xs = np.arange(len(losses))
            self.loss_line.set_data(xs, losses)
            self.axes[0].relim()
            self.axes[0].autoscale_view()

        #self.im_sim = self.axes[1].imshow(dummy_img, cmap="viridis")
        #self.im_diff = self.axes[2].imshow(dummy_img, cmap="viridis")
        #self.im_tgt = self.axes[3].imshow(self.target.detach().cpu(), cmap="viridis")
        #self.im_pots = [self.axes[i].imshow(dummy_img, cmap="viridis") for i in range(4, 8)]
        #self.im_ctf = self.axes[8].imshow(np.zeros((10, 10)), cmap="RdBu", vmin=-np.pi, vmax=np.pi)
        #self.im_ring = self.axes[8].imshow(np.zeros((10, 10)), cmap="Greens", alpha=0.4, origin="lower")


        # Images
        if sim is not None:
            #self.axes[1].imshow(sim, cmap="viridis")
            self.im_sim.set_data(sim)
        if diff is not None:
            self.im_diff.set_data(diff)
            #self.axes[2].imshow(diff, cmap="viridis")

        if tgt is not None:
            #self.axes[3].imshow(tgt, cmap="viridis")
            self.im_tgt.set_data(tgt)


        if P_diff is not None:
            self.axes[10].clear()
            self.axes[10].imshow(P_diff[0,:,:], aspect='auto', origin='lower', cmap='viridis')
            #self.axes[10].plot(np.linspace(0,1, P_diff.shape[1]), P_diff[0,:])
            self.axes[10].set_title("Diff Power Spectrum")

        if intensities is not None:
            self.axes[9].clear()
            n0,_,_ = self.axes[9].hist(intensities[0].detach().cpu().numpy(), bins=100, alpha=0.5, label='Graphen')
            n1,_,_ = self.axes[9].hist(intensities[1].detach().cpu().numpy(), bins=100, alpha=0.5, label='S low')
            n2,_,_ = self.axes[9].hist(intensities[2].detach().cpu().numpy(), bins=100, alpha=0.5, label='W')
            n3,_,_ = self.axes[9].hist(intensities[3].detach().cpu().numpy(), bins=100, alpha=0.5, label='S high')
            self.axes[9].set_title("Intensities")

            max_t = max(n0.max(), n1.max(), n2.max(), n3.max())

            self.axes[9].set_ylim(0, max_t)  # auto-scale y-axis
            self.axes[9].legend()

        self.pots_scatter = []
        for i in range(4):
            self.axes[i + 4].clear()
            pos, ints = self.pot_layer.get_pos_and_ints(i)
            self.pots_scatter.append(self.axes[i + 4].scatter(pos[0], pos[1], c=ints, s=2))

        #for i in range(4):
        #    if pots[i] is not None:
        #        self.im_pots[i].set_data(pots[i])
                #self.axes[i + 4].imshow(pots[i], cmap="viridis")

        if ctf_phase is not None:
            self.im_ctf.set_data(ctf_phase)
        if ring is not None:
            self.im_ring.set_data(ring)

        # Titles with step and some params
        try:
            c10 = self.ctf_layer.aberr_params["C10"].detach().cpu().numpy()
        except Exception:
            c10 = None
        try:
            c30 = self.ctf_layer.aberr_params["C30"].detach().cpu().numpy()
        except Exception:
            c30 = None

        self.axes[1].set_title(f"Simulated (Step {step})")
        self.axes[2].set_title(f"Diff (Step {step})")
        self.axes[3].set_title(f"Target (Step {step})")

        if c10 is not None or c30 is not None:
            self.axes[8].set_title(
                f"CTF (Step {step})\n"
                f"C10: {float(c10):.2f} nm   C30: {float(c30):.2f} um"
            )
        else:
            self.axes[8].set_title(f"CTF (Step {step})")

        self.canvas.draw_idle()

        # Reschedule
        self._schedule_refresh()

    def init_images(self):
        with self.viz.lock:
            sim = self.viz.sim_img
            diff = self.viz.diff_img
            tgt = self.viz.target_img
            pots = list(self.viz.pot_imgs) if self.viz.pot_imgs is not None else [None]*4
            ctf_phase = self.viz.ctf_phase
            ring = self.viz.ring_mask
            intensities = self.viz.intensities
        self.im_sim = self.axes[1].imshow(sim, cmap="viridis", origin="lower")
        self.im_diff = self.axes[2].imshow(diff, cmap="viridis", origin="lower")
        self.im_tgt = self.axes[3].imshow(tgt, cmap="viridis", origin="lower")
        self.pots_scatter = []
        for i in range(4):
                pos, ints = self.pot_layer.get_pos_and_ints(i)
                self.pots_scatter.append(self.axes[i + 4].scatter(pos[0], pos[1], c=ints))

        #self.im_pots = [self.axes[i].imshow(pots[i-4], cmap="viridis") for i in range(4, 8)]
        self.im_ctf = self.axes[8].imshow(ctf_phase, cmap="RdBu", vmin=-np.pi, vmax=np.pi)
        self.im_ring = self.axes[8].imshow(ring, cmap="Greens", alpha=0.4, origin="lower")

        self.axes[9].hist(intensities[0].detach().cpu().numpy(), bins=100, alpha=0.5, label='Graphen')
        self.axes[9].hist(intensities[1].detach().cpu().numpy(), bins=100, alpha=0.5, label='S low')
        self.axes[9].hist(intensities[2].detach().cpu().numpy(), bins=100, alpha=0.5, label='W')
        self.axes[9].hist(intensities[3].detach().cpu().numpy(), bins=100, alpha=0.5, label='S high')
        self.axes[9].legend()





# -------------------------
# Usage: plug in your objects
# -------------------------
def launch_gui(model, optimizer, target, brightness, contrast, pot_layer, ctf_layer):
    root = tk.Tk()
    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)

    app = LiveTrainGUI(
        root=root,
        model=model,
        optimizer=optimizer,
        target=target,
        brightness=brightness,
        contrast=contrast,
        pot_layer=pot_layer,
        ctf_layer=ctf_layer,
        plot_every=5,
        max_steps=2_000_000,
    )

    root.mainloop()