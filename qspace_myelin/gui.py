#!/usr/bin/env python3
"""
qspace-myelin — Graphical Interface
=====================================
PyQt6 desktop GUI for:
  - Tab 1: myelin_map.py       (NLD computation)
  - Tab 2: pipeline_myelin_map.sh  (full preprocessing + NLD)
  - Tab 3: visualize_nld.py    (publication-quality figures)

Requirements:
    pip install PyQt6

Usage:
    python gui_myelin_map.py
"""

__version__ = "0.1.0"

import os
import sys
import shutil
from pathlib import Path

from PyQt6.QtCore import (Qt, QProcess, QProcessEnvironment,
                           pyqtSignal, QObject)
from PyQt6.QtGui import (QFont, QColor, QTextCursor, QPixmap,
                          QIcon, QAction)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QComboBox,
    QDoubleSpinBox, QSpinBox, QCheckBox, QTextEdit,
    QFileDialog, QSplitter, QScrollArea, QGroupBox,
    QSizePolicy, QStatusBar, QMessageBox, QProgressBar,
    QFrame, QScrollArea,
)

# ── paths ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
MYELIN_SCRIPT      = _HERE / "myelin_map.py"
PIPELINE_SCRIPT    = _HERE / "pipeline_myelin_map.sh"
VISUALIZE_SCRIPT   = _HERE / "visualize_nld.py"

# ── palette ───────────────────────────────────────────────────────────────────
_STYLE = """
QMainWindow, QWidget { background: #1e1e2e; color: #cdd6f4; font-size: 13px; }
QTabWidget::pane { border: 1px solid #313244; border-radius: 6px; }
QTabBar::tab {
    background: #181825; color: #a6adc8; padding: 7px 20px;
    border-top-left-radius: 6px; border-top-right-radius: 6px;
    min-width: 160px;
}
QTabBar::tab:selected { background: #1e1e2e; color: #cdd6f4; border-bottom: 2px solid #89b4fa; }
QGroupBox {
    border: 1px solid #313244; border-radius: 6px;
    margin-top: 10px; padding-top: 6px; color: #89b4fa; font-weight: bold;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background: #313244; border: 1px solid #45475a;
    border-radius: 4px; padding: 4px 8px; color: #cdd6f4;
}
QLineEdit:focus, QComboBox:focus { border-color: #89b4fa; }
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
QComboBox:disabled { color: #585b70; background: #262637; }
QPushButton {
    background: #313244; color: #cdd6f4; border: 1px solid #45475a;
    border-radius: 4px; padding: 5px 14px; min-width: 80px;
}
QPushButton:hover { background: #45475a; }
QPushButton:pressed { background: #585b70; }
QPushButton#run_btn { background: #89b4fa; color: #1e1e2e; font-weight: bold; border: none; }
QPushButton#run_btn:hover { background: #b4d0ff; }
QPushButton#run_btn:disabled { background: #45475a; color: #585b70; }
QPushButton#stop_btn { background: #f38ba8; color: #1e1e2e; font-weight: bold; border: none; }
QPushButton#stop_btn:hover { background: #f4a8bb; }
QPushButton#stop_btn:disabled { background: #45475a; color: #585b70; }
QPushButton#browse_btn { min-width: 32px; padding: 4px 8px; font-size: 11px; }
QCheckBox { spacing: 6px; }
QCheckBox::indicator {
    width: 16px; height: 16px; border-radius: 3px;
    border: 1px solid #45475a; background: #313244;
}
QCheckBox::indicator:checked { background: #89b4fa; border-color: #89b4fa; }
QTextEdit {
    background: #11111b; color: #cdd6f4; border: 1px solid #313244;
    border-radius: 4px; font-family: monospace; font-size: 12px;
}
QScrollBar:vertical { background: #181825; width: 10px; border-radius: 5px; }
QScrollBar::handle:vertical { background: #45475a; border-radius: 5px; min-height: 24px; }
QProgressBar {
    background: #313244; border: none; border-radius: 4px;
    text-align: center; color: #cdd6f4; height: 6px;
}
QProgressBar::chunk { background: #89b4fa; border-radius: 4px; }
QStatusBar { background: #181825; color: #a6adc8; font-size: 11px; }
QLabel#section_label { color: #a6adc8; font-size: 11px; }
QSplitter::handle { background: #313244; }
"""


# ── reusable widgets ──────────────────────────────────────────────────────────

class FileField(QWidget):
    """A label + line-edit + browse button row for file/directory picking."""

    def __init__(self, label: str, mode: str = "file",
                 placeholder: str = "", filter_: str = "All files (*)",
                 optional: bool = False, parent=None):
        super().__init__(parent)
        self._mode = mode          # "file" | "save" | "dir"
        self._filter = filter_
        self._last_dir = str(Path.home())

        lbl = QLabel(label + ("  (optional)" if optional else ""))
        lbl.setFixedWidth(130)
        if optional:
            lbl.setObjectName("section_label")

        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)

        btn = QPushButton("…")
        btn.setObjectName("browse_btn")
        btn.setFixedWidth(32)
        btn.clicked.connect(self._browse)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(lbl)
        row.addWidget(self.edit)
        row.addWidget(btn)

    def _browse(self):
        start = self._last_dir
        if self._mode == "file":
            path, _ = QFileDialog.getOpenFileName(
                self, "Select file", start, self._filter)
        elif self._mode == "save":
            path, _ = QFileDialog.getSaveFileName(
                self, "Output prefix", start, self._filter)
        else:
            path = QFileDialog.getExistingDirectory(
                self, "Select directory", start)
        if path:
            self.edit.setText(path)
            self._last_dir = str(Path(path).parent)

    @property
    def value(self) -> str:
        return self.edit.text().strip()

    def set_value(self, v: str):
        self.edit.setText(v)


class LogWidget(QTextEdit):
    """Read-only log with coloured output."""

    _COLORS = {
        "err":  "#f38ba8",
        "warn": "#fab387",
        "ok":   "#a6e3a1",
        "info": "#cdd6f4",
        "dim":  "#585b70",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)

    def _append(self, text: str, key: str):
        color = self._COLORS.get(key, self._COLORS["info"])
        for line in text.splitlines():
            self.append(f'<span style="color:{color}">{line}</span>')
        self.moveCursor(QTextCursor.MoveOperation.End)

    def log_stdout(self, text: str):
        t = text.lower()
        if any(w in t for w in ("error", "✗", "traceback")):
            key = "err"
        elif any(w in t for w in ("warn", "⚠", "warning")):
            key = "warn"
        elif any(w in t for w in ("✓", "saved", "done", "complete", "ok")):
            key = "ok"
        elif text.startswith("  →") or text.startswith("  ["):
            key = "dim"
        else:
            key = "info"
        self._append(text, key)

    def log_stderr(self, text: str):
        self._append(text, "warn")

    def log_system(self, text: str):
        self._append(f"— {text} —", "dim")

    def clear_log(self):
        self.clear()


# ── process runner (shared by both tabs) ─────────────────────────────────────

class ProcessRunner(QObject):
    """Wraps QProcess; emits signals consumed by the tab that owns it."""

    stdout_line = pyqtSignal(str)
    stderr_line = pyqtSignal(str)
    finished    = pyqtSignal(int)   # exit code

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proc = QProcess(self)
        self._proc.readyReadStandardOutput.connect(self._read_stdout)
        self._proc.readyReadStandardError.connect(self._read_stderr)
        self._proc.finished.connect(lambda code, _: self.finished.emit(code))
        env = QProcessEnvironment.systemEnvironment()
        self._proc.setProcessEnvironment(env)

    def start(self, program: str, args: list[str]):
        self._proc.start(program, args)

    def stop(self):
        if self._proc.state() != QProcess.ProcessState.NotRunning:
            self._proc.kill()

    def is_running(self) -> bool:
        return self._proc.state() != QProcess.ProcessState.NotRunning

    def _read_stdout(self):
        raw = bytes(self._proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in raw.splitlines():
            self.stdout_line.emit(line)

    def _read_stderr(self):
        raw = bytes(self._proc.readAllStandardError()).decode("utf-8", errors="replace")
        for line in raw.splitlines():
            self.stderr_line.emit(line)


# ── image preview panel ───────────────────────────────────────────────────────

class ImagePreview(QScrollArea):
    """Shows a list of PNG images vertically."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setWidget(self._container)
        self.setMinimumHeight(200)

    def load_images(self, paths: list[str]):
        # Clear old images
        for i in reversed(range(self._layout.count())):
            w = self._layout.itemAt(i).widget()
            if w:
                w.deleteLater()

        for path in paths:
            if not Path(path).exists():
                continue
            lbl_title = QLabel(Path(path).name)
            lbl_title.setObjectName("section_label")
            self._layout.addWidget(lbl_title)

            lbl_img = QLabel()
            pix = QPixmap(path)
            if not pix.isNull():
                pix = pix.scaledToWidth(
                    560, Qt.TransformationMode.SmoothTransformation)
                lbl_img.setPixmap(pix)
            self._layout.addWidget(lbl_img)

            sep = QFrame()
            sep.setFrameShape(QFrame.Shape.HLine)
            sep.setStyleSheet("color: #313244;")
            self._layout.addWidget(sep)


# ── Tab 1: myelin_map.py ──────────────────────────────────────────────────────

class MyelinMapTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._runner = ProcessRunner(self)
        self._runner.stdout_line.connect(self._on_stdout)
        self._runner.stderr_line.connect(self._on_stderr)
        self._runner.finished.connect(self._on_finished)
        self._output_prefix = ""
        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # Left: controls
        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setMinimumWidth(420)
        ctrl_scroll.setMaximumWidth(520)
        ctrl_widget = QWidget()
        ctrl_scroll.setWidget(ctrl_widget)
        ctrl_layout = QVBoxLayout(ctrl_widget)
        ctrl_layout.setSpacing(10)

        # — Input files —
        grp_in = QGroupBox("Input files")
        g = QVBoxLayout(grp_in)
        self.f_dwi  = FileField("DWI (.nii.gz)",  filter_="NIfTI (*.nii *.nii.gz)")
        self.f_bval = FileField("bval",            filter_="bval files (*.bval);;All (*)")
        self.f_bvec = FileField("bvec",            filter_="bvec files (*.bvec);;All (*)")
        self.f_mask = FileField("Brain mask",      filter_="NIfTI (*.nii *.nii.gz)", optional=True)
        self.f_csf  = FileField("CSF mask",        filter_="NIfTI (*.nii *.nii.gz)", optional=True)
        for w in [self.f_dwi, self.f_bval, self.f_bvec, self.f_mask, self.f_csf]:
            g.addWidget(w)
        ctrl_layout.addWidget(grp_in)

        # — Normalization —
        grp_norm = QGroupBox("Normalization")
        gn = QFormLayout(grp_norm)
        gn.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.norm_mode = QComboBox()
        self.norm_mode.addItems(["auto", "csf", "internal", "manual"])
        self.norm_mode.setCurrentText("auto")
        self.norm_mode.currentTextChanged.connect(self._on_norm_changed)

        self.csf_pct = QSpinBox()
        self.csf_pct.setRange(50, 99); self.csf_pct.setValue(95)
        self.csf_pct.setSuffix("  (P%)")
        self.csf_pct.setToolTip(
            "Percentile for auto CSF detection.\n"
            "Lower = more voxels. Inspect *_csf_qc.png after running.")

        self.kmax = QDoubleSpinBox()
        self.kmax.setRange(-100, 1000); self.kmax.setValue(15.0); self.kmax.setDecimals(3)
        self.kmin = QDoubleSpinBox()
        self.kmin.setRange(-100, 1000); self.kmin.setValue(-2.0); self.kmin.setDecimals(3)

        gn.addRow("Mode:",    self.norm_mode)
        gn.addRow("CSF pct:", self.csf_pct)
        gn.addRow("Kmax:",    self.kmax)
        gn.addRow("Kmin:",    self.kmin)
        ctrl_layout.addWidget(grp_norm)
        self._on_norm_changed("auto")

        # — Processing —
        grp_proc = QGroupBox("Processing")
        gp = QFormLayout(grp_proc)
        gp.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.method = QComboBox()
        self.method.addItems(["fast (vectorized)", "slow (voxel-by-voxel)"])

        self.smooth = QDoubleSpinBox()
        self.smooth.setRange(0, 5); self.smooth.setValue(0.5); self.smooth.setSingleStep(0.1)
        self.smooth.setSuffix("  vox")
        self.smooth.setToolTip("Gaussian smoothing sigma in voxels (0 = disabled).")

        self.slice_axis = QComboBox()
        self.slice_axis.addItems(["Axial (z)", "Coronal (y)", "Sagittal (x)"])

        gp.addRow("Method:",     self.method)
        gp.addRow("Smoothing:",  self.smooth)
        gp.addRow("Slice view:", self.slice_axis)
        ctrl_layout.addWidget(grp_proc)

        # — Output —
        grp_out = QGroupBox("Output")
        go = QVBoxLayout(grp_out)
        self.f_out = FileField("Prefix", mode="save",
                               placeholder="e.g. /data/sub01/myelin")
        go.addWidget(self.f_out)
        ctrl_layout.addWidget(grp_out)

        # — Run controls —
        btn_row = QHBoxLayout()
        self.run_btn  = QPushButton("▶  Run")
        self.run_btn.setObjectName("run_btn")
        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._run)
        self.stop_btn.clicked.connect(self._stop)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.stop_btn)
        ctrl_layout.addLayout(btn_row)

        ctrl_layout.addStretch()
        ctrl_scroll.setWidget(ctrl_widget)

        # Right: log + preview
        right = QWidget()
        rl = QVBoxLayout(right)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)      # indeterminate
        self.progress.setVisible(False)
        rl.addWidget(self.progress)

        log_label = QLabel("Output log")
        log_label.setObjectName("section_label")
        rl.addWidget(log_label)
        self.log = LogWidget()
        rl.addWidget(self.log, stretch=2)

        prev_label = QLabel("Image preview  (updated after run)")
        prev_label.setObjectName("section_label")
        rl.addWidget(prev_label)
        self.preview = ImagePreview()
        rl.addWidget(self.preview, stretch=3)

        splitter.addWidget(ctrl_scroll)
        splitter.addWidget(right)
        splitter.setSizes([440, 760])

    # ── slot helpers ──────────────────────────────────────────────────────────

    def _on_norm_changed(self, mode: str):
        csf_modes = mode in ("csf", "internal")
        man_mode   = mode == "manual"
        self.csf_pct.setEnabled(csf_modes)
        self.kmax.setEnabled(man_mode)
        self.kmin.setEnabled(man_mode)

    def _on_stdout(self, line: str):
        self.log.log_stdout(line)

    def _on_stderr(self, line: str):
        self.log.log_stderr(line)

    def _on_finished(self, code: int):
        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if code == 0:
            self.log.log_system("Finished successfully")
            self._load_preview()
            self.window().statusBar().showMessage("Myelin map complete.", 8000)
        else:
            self.log.log_system(f"Process exited with code {code}")
            self.window().statusBar().showMessage(
                f"Process exited with code {code}.", 8000)

    def _load_preview(self):
        if not self._output_prefix:
            return
        candidates = [
            f"{self._output_prefix}_nld.png",
            f"{self._output_prefix}_nld_gray.png",
            f"{self._output_prefix}_csf_qc.png",
        ]
        self.preview.load_images(candidates)

    # ── run ───────────────────────────────────────────────────────────────────

    def _build_args(self) -> tuple[str, list[str]] | None:
        """Validate fields and build (program, args). Returns None on error."""

        errors = []
        if not self.f_dwi.value:  errors.append("DWI file is required.")
        if not self.f_bval.value: errors.append("bval file is required.")
        if not self.f_bvec.value: errors.append("bvec file is required.")
        if not self.f_out.value:  errors.append("Output prefix is required.")

        for lbl, field in [("DWI", self.f_dwi), ("bval", self.f_bval),
                            ("bvec", self.f_bvec)]:
            if field.value and not Path(field.value).exists():
                errors.append(f"{lbl} file not found: {field.value}")

        if errors:
            QMessageBox.critical(self, "Missing inputs", "\n".join(errors))
            return None

        script = MYELIN_SCRIPT
        if not script.exists():
            QMessageBox.critical(
                self, "Script not found",
                f"myelin_map.py not found at:\n{script}\n\n"
                "Place it in the same directory as this GUI.")
            return None

        mode = self.norm_mode.currentText()
        axis_map = {"Axial (z)": "2", "Coronal (y)": "1", "Sagittal (x)": "0"}
        meth_map = {"fast (vectorized)": "fast", "slow (voxel-by-voxel)": "slow"}

        args = [
            str(script),
            "--dwi",    self.f_dwi.value,
            "--bval",   self.f_bval.value,
            "--bvec",   self.f_bvec.value,
            "--output", self.f_out.value,
            "--norm-mode", mode,
            "--smooth",    str(self.smooth.value()),
            "--method",    meth_map[self.method.currentText()],
            "--slice-axis", axis_map[self.slice_axis.currentText()],
        ]

        if self.f_mask.value:
            args += ["--mask", self.f_mask.value]
        if self.f_csf.value:
            args += ["--csf-mask", self.f_csf.value]
        if mode in ("csf", "internal"):
            args += ["--csf-pct", str(self.csf_pct.value())]
        if mode == "manual":
            args += ["--kmax", str(self.kmax.value()),
                     "--kmin", str(self.kmin.value())]

        return sys.executable, args

    def _run(self):
        result = self._build_args()
        if result is None:
            return
        program, args = result
        self._output_prefix = self.f_out.value

        self.log.clear_log()
        self.log.log_system("Starting myelin_map.py")
        self.log.log_system("Command: " + " ".join([program] + args))

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setVisible(True)
        self.window().statusBar().showMessage("Running myelin map computation…")

        self._runner.start(program, args)

    def _stop(self):
        self._runner.stop()
        self.log.log_system("Process stopped by user.")
        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)


# ── Tab 2: full pipeline ──────────────────────────────────────────────────────

class PipelineTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._runner = ProcessRunner(self)
        self._runner.stdout_line.connect(self._on_stdout)
        self._runner.stderr_line.connect(self._on_stderr)
        self._runner.finished.connect(self._on_finished)
        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # Left: controls
        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setMinimumWidth(460)
        ctrl_scroll.setMaximumWidth(560)
        ctrl_widget = QWidget()
        ctrl_scroll.setWidget(ctrl_widget)
        ctrl_layout = QVBoxLayout(ctrl_widget)
        ctrl_layout.setSpacing(10)

        # — Input files —
        grp_in = QGroupBox("Input files")
        gi = QVBoxLayout(grp_in)
        self.f_dwi   = FileField("DWI (.nii.gz)",    filter_="NIfTI (*.nii *.nii.gz)")
        self.f_bval  = FileField("bval",              filter_="bval (*.bval);;All (*)")
        self.f_bvec  = FileField("bvec",              filter_="bvec (*.bvec);;All (*)")
        self.f_t1    = FileField("T1 (with skull)",   filter_="NIfTI (*.nii *.nii.gz)", optional=True)
        self.f_t1b   = FileField("T1 brain (no skull)", filter_="NIfTI (*.nii *.nii.gz)", optional=True)
        note = QLabel("  At least one T1 input is required.")
        note.setObjectName("section_label")
        self.f_csf   = FileField("CSF mask",          filter_="NIfTI (*.nii *.nii.gz)", optional=True)
        for w in [self.f_dwi, self.f_bval, self.f_bvec,
                  self.f_t1, self.f_t1b, note, self.f_csf]:
            gi.addWidget(w)
        ctrl_layout.addWidget(grp_in)

        # — Acquisition —
        grp_acq = QGroupBox("Acquisition")
        ga = QFormLayout(grp_acq)
        ga.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.pe_dir = QComboBox()
        self.pe_dir.addItems(["j-  (AP — anterior→posterior)",
                               "j   (PA — posterior→anterior)",
                               "i-  (RL — right→left)",
                               "i   (LR — left→right)"])
        self.readout = QDoubleSpinBox()
        self.readout.setRange(0.001, 1.0); self.readout.setValue(0.05)
        self.readout.setDecimals(4); self.readout.setSuffix("  s")
        self.readout.setToolTip("Total EPI readout time in seconds.")
        ga.addRow("PE direction:", self.pe_dir)
        ga.addRow("Readout time:", self.readout)
        ctrl_layout.addWidget(grp_acq)

        # — Processing —
        grp_proc = QGroupBox("Processing")
        gp = QFormLayout(grp_proc)
        gp.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.nthreads = QSpinBox()
        self.nthreads.setRange(1, 64); self.nthreads.setValue(4)
        self.bet_t1 = QDoubleSpinBox()
        self.bet_t1.setRange(0.1, 0.9); self.bet_t1.setValue(0.35)
        self.bet_t1.setSingleStep(0.05)
        self.bet_t1.setToolTip("BET fractional intensity threshold for T1.\n"
                                "Lower = more aggressive skull stripping.")
        self.bet_dwi = QDoubleSpinBox()
        self.bet_dwi.setRange(0.1, 0.9); self.bet_dwi.setValue(0.3)
        self.bet_dwi.setSingleStep(0.05)
        self.smooth = QDoubleSpinBox()
        self.smooth.setRange(0, 5); self.smooth.setValue(0.5)
        self.smooth.setSingleStep(0.1); self.smooth.setSuffix("  vox")
        self.skip_denoise = QCheckBox("Skip denoising  (dwidenoise)")
        self.skip_eddy    = QCheckBox("Skip eddy correction")
        gp.addRow("Threads:",      self.nthreads)
        gp.addRow("BET f (T1):",   self.bet_t1)
        gp.addRow("BET f (DWI):",  self.bet_dwi)
        gp.addRow("Smoothing:",    self.smooth)
        gp.addRow("",              self.skip_denoise)
        gp.addRow("",              self.skip_eddy)
        ctrl_layout.addWidget(grp_proc)

        # — Normalization —
        grp_norm = QGroupBox("NLD Normalization")
        gn = QFormLayout(grp_norm)
        gn.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.norm_mode = QComboBox()
        self.norm_mode.addItems(["auto", "csf", "internal", "manual"])
        self.norm_mode.currentTextChanged.connect(self._on_norm_changed)
        self.csf_pct = QSpinBox()
        self.csf_pct.setRange(50, 99); self.csf_pct.setValue(95)
        self.csf_pct.setSuffix("  (P%)")
        self.csf_pct.setToolTip("Percentile for auto CSF detection.\n"
                                  "Inspect csf_mask_qc.png in qc/ folder.")
        self.kmax = QDoubleSpinBox()
        self.kmax.setRange(-100, 1000); self.kmax.setValue(15.0); self.kmax.setDecimals(3)
        self.kmin = QDoubleSpinBox()
        self.kmin.setRange(-100, 1000); self.kmin.setValue(-2.0); self.kmin.setDecimals(3)
        gn.addRow("Mode:",    self.norm_mode)
        gn.addRow("CSF pct:", self.csf_pct)
        gn.addRow("Kmax:",    self.kmax)
        gn.addRow("Kmin:",    self.kmin)
        ctrl_layout.addWidget(grp_norm)
        self._on_norm_changed("auto")

        # — Output —
        grp_out = QGroupBox("Output")
        go = QVBoxLayout(grp_out)
        self.f_out = FileField("Output dir", mode="dir",
                               placeholder="e.g. /data/sub01/output_myelin")
        go.addWidget(self.f_out)
        ctrl_layout.addWidget(grp_out)

        # — Run controls —
        btn_row = QHBoxLayout()
        self.run_btn  = QPushButton("▶  Run pipeline")
        self.run_btn.setObjectName("run_btn")
        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._run)
        self.stop_btn.clicked.connect(self._stop)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.stop_btn)
        ctrl_layout.addLayout(btn_row)
        ctrl_layout.addStretch()

        # Right: log
        right = QWidget()
        rl = QVBoxLayout(right)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        rl.addWidget(self.progress)
        log_label = QLabel("Pipeline log")
        log_label.setObjectName("section_label")
        rl.addWidget(log_label)
        self.log = LogWidget()
        rl.addWidget(self.log)

        splitter.addWidget(ctrl_scroll)
        splitter.addWidget(right)
        splitter.setSizes([480, 720])

    # ── slots ─────────────────────────────────────────────────────────────────

    def _on_norm_changed(self, mode: str):
        csf_modes = mode in ("csf", "internal")
        self.csf_pct.setEnabled(csf_modes)
        self.kmax.setEnabled(mode == "manual")
        self.kmin.setEnabled(mode == "manual")

    def _on_stdout(self, line: str):
        self.log.log_stdout(line)

    def _on_stderr(self, line: str):
        self.log.log_stderr(line)

    def _on_finished(self, code: int):
        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if code == 0:
            self.log.log_system("Pipeline finished successfully")
            self.window().statusBar().showMessage("Pipeline complete.", 8000)
        else:
            self.log.log_system(f"Process exited with code {code}")
            self.window().statusBar().showMessage(
                f"Pipeline exited with code {code}.", 8000)

    # ── run ───────────────────────────────────────────────────────────────────

    def _build_args(self) -> tuple[str, list[str]] | None:
        errors = []
        if not self.f_dwi.value:  errors.append("DWI file is required.")
        if not self.f_bval.value: errors.append("bval file is required.")
        if not self.f_bvec.value: errors.append("bvec file is required.")
        if not self.f_t1.value and not self.f_t1b.value:
            errors.append("At least one T1 input (--t1 or --t1brain) is required.")
        if not self.f_out.value:  errors.append("Output directory is required.")

        for lbl, field in [("DWI", self.f_dwi), ("bval", self.f_bval),
                            ("bvec", self.f_bvec)]:
            if field.value and not Path(field.value).exists():
                errors.append(f"{lbl} file not found: {field.value}")

        if errors:
            QMessageBox.critical(self, "Missing inputs", "\n".join(errors))
            return None

        bash = shutil.which("bash")
        if not bash:
            QMessageBox.critical(self, "bash not found",
                                  "bash is required to run the pipeline.")
            return None

        script = PIPELINE_SCRIPT
        if not script.exists():
            QMessageBox.critical(
                self, "Script not found",
                f"pipeline_myelin_map.sh not found at:\n{script}")
            return None

        pe_map = {"j-": "j-", "j": "j", "i-": "i-", "i": "i"}
        pe_raw = self.pe_dir.currentText().split()[0]

        mode = self.norm_mode.currentText()
        args = [
            str(script),
            "--dwi",         self.f_dwi.value,
            "--bval",        self.f_bval.value,
            "--bvec",        self.f_bvec.value,
            "--output_dir",  self.f_out.value,
            "--pe_dir",      pe_raw,
            "--readout",     str(self.readout.value()),
            "--nthreads",    str(self.nthreads.value()),
            "--bet_f_t1",    str(self.bet_t1.value()),
            "--bet_f_dwi",   str(self.bet_dwi.value()),
            "--smooth",      str(self.smooth.value()),
            "--norm_mode",   mode,
            "--csf_pct",     str(self.csf_pct.value()),
        ]

        if self.f_t1.value:   args += ["--t1",      self.f_t1.value]
        if self.f_t1b.value:  args += ["--t1brain",  self.f_t1b.value]
        if self.f_csf.value:  args += ["--csf_mask", self.f_csf.value]
        if mode == "manual":
            args += ["--kmax", str(self.kmax.value()),
                     "--kmin", str(self.kmin.value())]
        if self.skip_denoise.isChecked(): args.append("--skip_denoise")
        if self.skip_eddy.isChecked():    args.append("--skip_eddy")

        return bash, args

    def _run(self):
        result = self._build_args()
        if result is None:
            return
        program, args = result

        self.log.clear_log()
        self.log.log_system("Starting pipeline")
        self.log.log_system("Command: " + " ".join([program] + args))

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setVisible(True)
        self.window().statusBar().showMessage("Pipeline running…")

        self._runner.start(program, args)

    def _stop(self):
        self._runner.stop()
        self.log.log_system("Pipeline stopped by user.")
        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)


# ── Tab 3: visualization ─────────────────────────────────────────────────────

class VisualizationTab(QWidget):
    """Controls for visualize_nld.py — generates publication-quality figures."""

    _CMAPS = ["hot", "paper", "thermal", "plasma", "inferno", "viridis"]
    _CMAP_DESC = {
        "hot":     "Black → red → yellow  (high contrast)",
        "paper":   "White → blue  (Fujiyoshi et al. style)",
        "thermal": "Black → blue → cyan → yellow → white",
        "plasma":  "Purple → magenta → yellow",
        "inferno": "Black → purple → orange  (colourblind-safe)",
        "viridis": "Purple → teal → yellow  (perceptually uniform)",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._runner = ProcessRunner(self)
        self._runner.stdout_line.connect(self._on_stdout)
        self._runner.stderr_line.connect(self._on_stderr)
        self._runner.finished.connect(self._on_finished)
        self._output_prefix = ""
        self._build_ui()

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ── Left: controls ────────────────────────────────────────────────
        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setMinimumWidth(400)
        ctrl_scroll.setMaximumWidth(500)
        ctrl_widget = QWidget()
        ctrl_scroll.setWidget(ctrl_widget)
        ctrl = QVBoxLayout(ctrl_widget)
        ctrl.setSpacing(10)

        # — Input files —
        grp_in = QGroupBox("Input files")
        gi = QVBoxLayout(grp_in)
        self.f_nld  = FileField("NLD map",    filter_="NIfTI (*.nii *.nii.gz)")
        self.f_t1   = FileField("T1 brain",   filter_="NIfTI (*.nii *.nii.gz)", optional=True)
        self.f_mask = FileField("Brain mask", filter_="NIfTI (*.nii *.nii.gz)", optional=True)
        for w in [self.f_nld, self.f_t1, self.f_mask]:
            gi.addWidget(w)
        ctrl.addWidget(grp_in)

        # — Colormap —
        grp_cmap = QGroupBox("Colormap")
        gc = QVBoxLayout(grp_cmap)
        cmap_row = QHBoxLayout()
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(self._CMAPS)
        self.cmap_combo.setCurrentText("hot")
        self.cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        self._cmap_desc = QLabel(self._CMAP_DESC["hot"])
        self._cmap_desc.setObjectName("section_label")
        self._cmap_desc.setWordWrap(True)
        cmap_row.addWidget(QLabel("Colormap:"))
        cmap_row.addWidget(self.cmap_combo, stretch=1)
        gc.addLayout(cmap_row)
        gc.addWidget(self._cmap_desc)
        ctrl.addWidget(grp_cmap)

        # — Display window —
        grp_win = QGroupBox("Display window  (NLD)")
        gw = QFormLayout(grp_win)
        gw.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.vmin = QDoubleSpinBox()
        self.vmin.setRange(0, 100); self.vmin.setValue(10)
        self.vmin.setSuffix("  NLD"); self.vmin.setDecimals(1)
        self.vmin.setToolTip("Lower NLD display bound. Voxels below this value are transparent.")

        self.vmax = QDoubleSpinBox()
        self.vmax.setRange(0, 100); self.vmax.setValue(95)
        self.vmax.setSuffix("  NLD"); self.vmax.setDecimals(1)
        self.vmax.setToolTip("Upper NLD display bound.")

        self.alpha = QDoubleSpinBox()
        self.alpha.setRange(0.1, 1.0); self.alpha.setValue(0.80)
        self.alpha.setSingleStep(0.05); self.alpha.setDecimals(2)
        self.alpha.setToolTip("Overlay opacity over the T1 background (1 = opaque).")

        self.auto_window = QCheckBox("Auto window  (P2–P98 from data)")
        self.auto_window.setChecked(True)
        self.auto_window.toggled.connect(self._on_auto_window)

        gw.addRow("",        self.auto_window)
        gw.addRow("V min:",  self.vmin)
        gw.addRow("V max:",  self.vmax)
        gw.addRow("Alpha:",  self.alpha)
        self._on_auto_window(True)
        ctrl.addWidget(grp_win)

        # — Figures to generate —
        grp_fig = QGroupBox("Figures to generate")
        gf = QVBoxLayout(grp_fig)
        self.chk_montage   = QCheckBox("Montage  — multi-slice axial grid")
        self.chk_triplane  = QCheckBox("Triplane  — axial + coronal + sagittal")
        self.chk_mosaic    = QCheckBox("Mosaic  — all slices overview")
        self.chk_histogram = QCheckBox("Histogram  — NLD distribution")
        for chk in [self.chk_montage, self.chk_triplane,
                    self.chk_mosaic, self.chk_histogram]:
            chk.setChecked(True)
            gf.addWidget(chk)
        ctrl.addWidget(grp_fig)

        # — Options —
        grp_opt = QGroupBox("Options")
        go = QFormLayout(grp_opt)
        go.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.ncols = QSpinBox()
        self.ncols.setRange(3, 12); self.ncols.setValue(6)
        self.ncols.setToolTip("Number of columns in the montage grid.")

        self.dpi = QSpinBox()
        self.dpi.setRange(72, 600); self.dpi.setValue(300)
        self.dpi.setSuffix("  dpi")

        self.axis = QComboBox()
        self.axis.addItems(["Axial (z)", "Coronal (y)", "Sagittal (x)"])

        go.addRow("Montage cols:", self.ncols)
        go.addRow("Resolution:",   self.dpi)
        go.addRow("Slice axis:",   self.axis)
        ctrl.addWidget(grp_opt)

        # — Output —
        grp_out = QGroupBox("Output")
        gout = QVBoxLayout(grp_out)
        self.f_out = FileField("Prefix", mode="save",
                               placeholder="e.g. /data/figures/sub01")
        gout.addWidget(self.f_out)
        ctrl.addWidget(grp_out)

        # — Run controls —
        btn_row = QHBoxLayout()
        self.run_btn  = QPushButton("▶  Generate figures")
        self.run_btn.setObjectName("run_btn")
        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._run)
        self.stop_btn.clicked.connect(self._stop)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.stop_btn)
        ctrl.addLayout(btn_row)
        ctrl.addStretch()

        # ── Right: log + preview ──────────────────────────────────────────
        right = QWidget()
        rl = QVBoxLayout(right)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        rl.addWidget(self.progress)

        log_label = QLabel("Output log")
        log_label.setObjectName("section_label")
        rl.addWidget(log_label)
        self.log = LogWidget()
        self.log.setMaximumHeight(160)
        rl.addWidget(self.log)

        prev_label = QLabel("Preview  (click figures to refresh)")
        prev_label.setObjectName("section_label")
        rl.addWidget(prev_label)

        # Tab widget for figure previews
        self.preview_tabs = QTabWidget()
        self.preview_tabs.setStyleSheet(
            "QTabBar::tab { min-width: 100px; padding: 4px 12px; }")
        self._previews: dict[str, ImagePreview] = {}
        for name in ["montage", "triplane", "mosaic", "histogram"]:
            pv = ImagePreview()
            self.preview_tabs.addTab(pv, name.capitalize())
            self._previews[name] = pv
        rl.addWidget(self.preview_tabs, stretch=1)

        splitter.addWidget(ctrl_scroll)
        splitter.addWidget(right)
        splitter.setSizes([420, 860])

    # ── slots ─────────────────────────────────────────────────────────────

    def _on_cmap_changed(self, name: str):
        self._cmap_desc.setText(self._CMAP_DESC.get(name, ""))

    def _on_auto_window(self, checked: bool):
        self.vmin.setEnabled(not checked)
        self.vmax.setEnabled(not checked)

    def _on_stdout(self, line: str):
        self.log.log_stdout(line)
        # Auto-refresh preview when a file is saved
        if "Saved:" in line:
            path = line.split("Saved:")[-1].strip()
            for name in ["montage", "triplane", "mosaic", "histogram"]:
                if f"_{name}.png" in path:
                    self._previews[name].load_images([path])
                    idx = ["montage","triplane","mosaic","histogram"].index(name)
                    self.preview_tabs.setCurrentIndex(idx)

    def _on_stderr(self, line: str):
        self.log.log_stderr(line)

    def _on_finished(self, code: int):
        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if code == 0:
            self.log.log_system("Done.")
            self.window().statusBar().showMessage("Figures generated.", 8000)
        else:
            self.log.log_system(f"Exited with code {code}.")
            self.window().statusBar().showMessage(f"Error (code {code}).", 8000)

    # ── run ───────────────────────────────────────────────────────────────

    def _build_args(self) -> tuple[str, list[str]] | None:
        errors = []
        if not self.f_nld.value: errors.append("NLD map is required.")
        if not self.f_out.value: errors.append("Output prefix is required.")
        if self.f_nld.value and not Path(self.f_nld.value).exists():
            errors.append(f"NLD file not found: {self.f_nld.value}")

        if errors:
            QMessageBox.critical(self, "Missing inputs", "\n".join(errors))
            return None

        if not VISUALIZE_SCRIPT.exists():
            QMessageBox.critical(
                self, "Script not found",
                f"visualize_nld.py not found at:\n{VISUALIZE_SCRIPT}")
            return None

        axis_map = {"Axial (z)": "2", "Coronal (y)": "1", "Sagittal (x)": "0"}

        args = [
            str(VISUALIZE_SCRIPT),
            "--nld",    self.f_nld.value,
            "--output", self.f_out.value,
            "--cmap",   self.cmap_combo.currentText(),
            "--alpha",  str(self.alpha.value()),
            "--ncols",  str(self.ncols.value()),
            "--dpi",    str(self.dpi.value()),
            "--axis",   axis_map[self.axis.currentText()],
        ]

        if self.f_t1.value:   args += ["--t1",   self.f_t1.value]
        if self.f_mask.value: args += ["--mask",  self.f_mask.value]

        if not self.auto_window.isChecked():
            args += ["--vmin", str(self.vmin.value()),
                     "--vmax", str(self.vmax.value())]

        if not self.chk_montage.isChecked():   args.append("--no-montage")
        if not self.chk_triplane.isChecked():  args.append("--no-triplane")
        if not self.chk_mosaic.isChecked():    args.append("--no-mosaic")
        if not self.chk_histogram.isChecked(): args.append("--no-histogram")

        return sys.executable, args

    def _run(self):
        result = self._build_args()
        if result is None:
            return
        program, args = result
        self._output_prefix = self.f_out.value

        self.log.clear_log()
        self.log.log_system("Starting visualization")

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setVisible(True)
        self.window().statusBar().showMessage("Generating figures…")

        self._runner.start(program, args)

    def _stop(self):
        self._runner.stop()
        self.log.log_system("Stopped by user.")
        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)


# ── main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"qspace-myelin  v{__version__}")
        self.resize(1280, 820)
        self.setMinimumSize(900, 600)

        tabs = QTabWidget()
        tabs.addTab(MyelinMapTab(),     "  NLD Myelin Map  ")
        tabs.addTab(PipelineTab(),      "  Full Pipeline   ")
        tabs.addTab(VisualizationTab(), "  Visualization   ")
        self.setCentralWidget(tabs)

        bar = QStatusBar()
        self.setStatusBar(bar)
        bar.showMessage("Ready.")

        self._build_menu()

    def _build_menu(self):
        menu = self.menuBar()

        m_file = menu.addMenu("File")
        act_quit = QAction("Quit", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        m_file.addAction(act_quit)

        m_help = menu.addMenu("Help")
        act_about = QAction("About", self)
        act_about.triggered.connect(self._about)
        m_help.addAction(act_about)

    def _about(self):
        QMessageBox.about(
            self, "About qspace-myelin",
            f"<b>qspace-myelin</b> v{__version__}<br><br>"
            "NLD Myelin Map — q-space diffusion MRI<br>"
            "Based on: Fujiyoshi et al., <i>J. Neurosci.</i>, 2016, "
            "36(9):2796–2808<br><br>"
            "GPL-3.0 License"
        )


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(_STYLE)
    app.setApplicationName("qspace-myelin")
    app.setApplicationVersion(__version__)

    font = QFont()
    font.setPointSize(11)
    app.setFont(font)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
