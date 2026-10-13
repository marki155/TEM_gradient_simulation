import tkinter as tk
from tkinter import ttk
from dataclasses import fields, is_dataclass


class TrainMethodEditor:
    def __init__(self, parent, train_method, callback=None):
        if not is_dataclass(train_method):
            raise TypeError("train_method muss eine Dataclass-Instanz sein")

        self.parent = parent
        self.train_method = train_method
        self.callback = callback

        # hier speichern wir die tkinter-Variablen pro Attributname
        self.tk_vars = {}

        self.frame = ttk.LabelFrame(parent, text="TrainMethod", padding=8)
        self.frame.grid_columnconfigure(1, weight=1)

        self._build_from_dataclass()

    def _build_from_dataclass(self):
        row = 0

        for f in fields(self.train_method):
            name = f.name
            value = getattr(self.train_method, name)

            ttk.Label(self.frame, text=name).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)

            # bool -> checkbox
            if isinstance(value, bool):
                var = tk.BooleanVar(value=value)
                cb = ttk.Checkbutton(
                    self.frame,
                    variable=var,
                    command=lambda n=name, v=var: self._update_attr(n, v.get())
                )
                cb.grid(row=row, column=1, sticky="w", pady=2)
                self.tk_vars[name] = var

            # str -> combobox (für bekannte Attribute / bekannte Werte)
            elif isinstance(value, str):
                if name == "loss_fkt":
                    options = ["MSE", "NCC", "SSIM"]
                    var = tk.StringVar(value=value)
                    combo = ttk.Combobox(
                        self.frame,
                        textvariable=var,
                        values=options,
                        state="readonly"
                    )
                    combo.grid(row=row, column=1, sticky="ew", pady=2)
                    combo.bind(
                        "<<ComboboxSelected>>",
                        lambda event, n=name, v=var: self._update_attr(n, v.get())
                    )
                    self.tk_vars[name] = var
                else:
                    var = tk.StringVar(value=value)
                    entry = ttk.Entry(self.frame, textvariable=var)
                    entry.grid(row=row, column=1, sticky="ew", pady=2)
                    entry.bind(
                        "<KeyRelease>",
                        lambda event, n=name, v=var: self._update_attr(n, v.get())
                    )
                    self.tk_vars[name] = var

            else:
                # Fallback: als Textfeld anzeigen
                var = tk.StringVar(value=str(value))
                entry = ttk.Entry(self.frame, textvariable=var)
                entry.grid(row=row, column=1, sticky="ew", pady=2)
                entry.bind(
                    "<KeyRelease>",
                    lambda event, n=name, v=var: self._update_attr(n, v.get())
                )
                self.tk_vars[name] = var

            row += 1

    def graphen_training(self):
        self.train_method.learing_rate = 1.e-9

        self.train_method.train_graphen_intensities = False
        self.train_method.train_S1_intensities = False
        self.train_method.train_W_intensities = False
        self.train_method.train_S2_intensities = False

        self.train_method.train_graphen_occupancies = False
        self.train_method.train_S1_occupancies = False
        self.train_method.train_W_occupancies = False
        self.train_method.train_S2_occupancies = False

        self.train_method.graphen_shift = True
        self.train_method.graphen_rotation = True
        self.train_method.graphen_free_positions = False
        self.train_method.graphen_scale = True
        self.train_method.WS2_shift = False
        self.train_method.WS2_rotation = False
        self.train_method.WS2_free_positions = False
        self.train_method.WS2_scale = False
        self.train_method.train_ctf_aberrations = False
        self.train_method.train_multislice_sigma = False

        self.train_method.only_train_small_area = 0

        self.refresh_from_object()

    def WS2_training(self):

        self.train_method.train_blur = True
        self.train_method.train_amplitude_contrast = True

        self.train_method.learing_rate = 0.00001
        self.train_method.train_graphen_intensities = False
        self.train_method.train_S1_intensities = False
        self.train_method.train_W_intensities = False
        self.train_method.train_S2_intensities = False

        self.train_method.train_graphen_occupancies = False
        self.train_method.train_S1_occupancies = False
        self.train_method.train_W_occupancies = False
        self.train_method.train_S2_occupancies = False

        self.train_method.graphen_shift = False
        self.train_method.graphen_rotation = False
        self.train_method.graphen_free_positions = False
        self.train_method.graphen_scale = False
        self.train_method.WS2_shift = True
        self.train_method.WS2_rotation = True
        self.train_method.WS2_free_positions = False
        self.train_method.WS2_scale = True
        self.train_method.train_ctf_aberrations = True
        self.train_method.train_multislice_sigma = True

        self.train_method.train_env_parameters = True

        self.train_method.only_train_small_area = 1

        self.refresh_from_object()

    def main_training(self):
        self.train_method.learing_rate = 0.0001

        self.train_method.train_blur = False
        self.train_method.train_amplitude_contrast = False

        self.train_method.train_graphen_intensities = False
        self.train_method.train_S1_intensities = False
        self.train_method.train_W_intensities = False
        self.train_method.train_S2_intensities = False

        self.train_method.train_graphen_occupancies = False
        self.train_method.train_S1_occupancies = True
        self.train_method.train_W_occupancies = True
        self.train_method.train_S2_occupancies = True

        self.train_method.graphen_shift = False
        self.train_method.graphen_rotation = False
        self.train_method.graphen_free_positions = False
        self.train_method.graphen_scale = False
        self.train_method.WS2_shift = False
        self.train_method.WS2_rotation = False
        self.train_method.WS2_free_positions = True
        self.train_method.WS2_scale = False
        self.train_method.train_ctf_aberrations = False
        self.train_method.train_multislice_sigma = False

        self.train_method.only_train_small_area = -1

        self.refresh_from_object()


    def _update_attr(self, name, value):
        setattr(self.train_method, name, value)
        if self.callback is not None:
            self.callback()

    def refresh_from_object(self):
        """
        Falls train_method extern geändert wurde, GUI wieder synchronisieren.
        """
        for f in fields(self.train_method):
            name = f.name
            if name in self.tk_vars:
                self.tk_vars[name].set(getattr(self.train_method, name))