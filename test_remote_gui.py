import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QTextEdit,
    QSlider, QCheckBox
)
from PySide6.QtCore import Qt


class TestGui(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Qt Test-GUI")
        self.resize(500, 400)

        layout = QVBoxLayout()

        self.title_label = QLabel("Test-GUI auf dem Server")
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)

        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Text eingeben...")
        self.button = QPushButton("Anzeigen")
        self.button.clicked.connect(self.on_button_clicked)

        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.button)
        layout.addLayout(input_layout)

        self.result_label = QLabel("Noch keine Eingabe")
        layout.addWidget(self.result_label)

        self.checkbox = QCheckBox("Großbuchstaben verwenden")
        layout.addWidget(self.checkbox)

        slider_label = QLabel("Wert:")
        layout.addWidget(slider_label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(50)
        self.slider.valueChanged.connect(self.on_slider_changed)
        layout.addWidget(self.slider)

        self.slider_value_label = QLabel("Slider-Wert: 50")
        layout.addWidget(self.slider_value_label)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.append("GUI gestartet.")
        layout.addWidget(self.log_output)

        self.setLayout(layout)

    def on_button_clicked(self):
        text = self.input_field.text().strip()
        if not text:
            text = "(leer)"

        if self.checkbox.isChecked():
            text = text.upper()

        self.result_label.setText(f"Eingabe: {text}")
        self.log_output.append(f"Button geklickt, Text: {text}")

    def on_slider_changed(self, value):
        self.slider_value_label.setText(f"Slider-Wert: {value}")
        self.log_output.append(f"Slider geändert auf: {value}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TestGui()
    window.show()
    sys.exit(app.exec())
