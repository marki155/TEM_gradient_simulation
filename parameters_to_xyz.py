
from PySide6.QtWidgets import QFileDialog


def load_parameters():
    file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Parameter laden",
                "parameters",
                "JSON-Dateien (*.json)"
            )

            if not file_path:
                return

            loaded = self.load_dataclass_json(file_path, type(self.params_root))
            self.params_root = loaded
            return

            if selected_fields is None:
                self.params_root = loaded
                return

            for field_name in selected_fields:
                if hasattr(loaded, field_name) and hasattr(self.params_root, field_name):
                    setattr(self.params_root, field_name, getattr(loaded, field_name))