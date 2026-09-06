import json
import os
import shutil
import sys
import numpy as np
import soundfile as sf
from pathlib import Path

from PySide6.QtCore import QElapsedTimer, QPointF, QProcess, QRectF, Qt, QThread, QTimer, Signal, QUrl
from PySide6.QtGui import QDesktopServices, QColor, QPainter, QPen
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
    QInputDialog, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QScrollArea, QSlider, QStyle, QStyleOptionSlider, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
from engine import EXTENSIONS, STEMS, SPECIALISTS, Separation
from analysis.drum_midi import LANES, load_or_estimate
from analysis.piano_midi import load_or_estimate as load_or_estimate_piano
from analysis.bass_midi import BASS_LOW, BASS_HIGH, load_or_estimate as load_or_estimate_bass
from analysis.vocal_midi import VOCAL_LOW, VOCAL_HIGH, load_or_estimate as load_or_estimate_vocal
from analysis.chord_estimator import load_or_estimate as load_or_estimate_chords
from analysis.acoustic_chords import CACHE_FILE as ACOUSTIC_CHORD_CACHE, load_or_estimate as load_or_estimate_acoustic, prepare_chords
from analysis.beat_grid import load_or_estimate as load_or_estimate_beats, quantize_drum_events, quantize_note_events

ROOT = Path(__file__).resolve().parent
USER_CHORD_SHAPES_PATH = ROOT / "chord_shape.json"
NAMES = {"vocals": "ボーカル", "drums": "ドラム", "bass": "ベース",
         "other": "その他", "guitar": "エレキ + アコースティックギター", "piano": "ピアノ"}
NAMES["acoustic-guitar"] = "アコースティックギター"
NAMES["guitar-other"] = "エレキギター"
COLORS = {"vocals": "#c0abff", "drums": "#ffbc8a", "bass": "#87d7c3",
          "other": "#8bbaf6", "guitar": "#e8a3c9", "piano": "#d5d98d"}
COLORS["acoustic-guitar"] = "#f1c27d"
COLORS["guitar-other"] = "#d9966a"
DRUM_NAMES = {"kick": "Kick", "snare": "Snare", "tom": "Tom", "hihat": "Hi-hat", "cymbal": "Cymbal"}
DRUM_COLORS = {"kick": "#90d7ff", "snare": "#ff9da8", "tom": "#d7a8ff", "hihat": "#ffe08a", "cymbal": "#cfe98a"}
DRUM_DISPLAY_LANES = ("cymbal", "hihat", "tom", "snare", "kick")
PIANO_LOW = 21
PIANO_HIGH = 108


def timestamp(milliseconds):
    seconds = int(max(0, milliseconds) // 1000)
    return f"{seconds // 60}:{seconds % 60:02d}"


class SeekSlider(QSlider):
    """Jump to a clicked position, then allow normal dragging and arrow keys."""
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            option = QStyleOptionSlider()
            self.initStyleOption(option)
            handle = self.style().subControlRect(QStyle.CC_Slider, option, QStyle.SC_SliderHandle, self)
            if not handle.contains(event.position().toPoint()):
                self.setSliderDown(True)
                self.setValue(QStyle.sliderValueFromPosition(
                    self.minimum(), self.maximum(),
                    round(event.position().x() - handle.width() / 2),
                    max(1, self.width() - handle.width()), option.upsideDown))
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.isSliderDown():
            option = QStyleOptionSlider()
            self.initStyleOption(option)
            handle = self.style().subControlRect(QStyle.CC_Slider, option, QStyle.SC_SliderHandle, self)
            self.setValue(QStyle.sliderValueFromPosition(
                self.minimum(), self.maximum(), round(event.position().x() - handle.width() / 2),
                max(1, self.width() - handle.width()), option.upsideDown))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.setSliderDown(False)


class DrumMidiView(QWidget):
    def __init__(self):
        super().__init__()
        self.events = {lane: [] for lane in LANES}
        self.beat_grid = {}
        self.position = 0.0
        self.window_seconds = 2.5
        self.setMinimumHeight(240)

    def set_events(self, events):
        self.events = {lane: list(events.get(lane, [])) for lane in LANES}
        self.update()

    def set_beat_grid(self, beat_grid):
        self.beat_grid = beat_grid or {}
        self.update()

    def set_position_ms(self, milliseconds):
        self.position = max(0.0, milliseconds / 1000)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(12, 12, -12, -12)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#1e212a"))
        painter.drawRoundedRect(rect, 8, 8)

        label_width = 82
        lane_height = rect.height() / len(DRUM_DISPLAY_LANES)
        start = max(0.0, self.position)
        end = start + self.window_seconds
        text_pen = QPen(QColor("#b7bac8"))
        play_x = rect.left() + label_width
        active_window = 0.08
        self.draw_beat_grid(painter, rect, label_width, start, end)

        for index, lane in enumerate(DRUM_DISPLAY_LANES):
            top = rect.top() + index * lane_height
            center_y = top + lane_height / 2
            active = any(abs(float(seconds) - self.position) <= active_window for seconds in self.events.get(lane, []))
            painter.setPen(QPen(QColor(DRUM_COLORS[lane])) if active else text_pen)
            font = painter.font()
            font.setPointSize(13)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(QRectF(rect.left() + 14, top, label_width - 18, lane_height), Qt.AlignVCenter, DRUM_NAMES[lane])
            painter.setPen(QPen(QColor("#30333d")))
            painter.drawLine(rect.left() + label_width, round(center_y), rect.right() - 8, round(center_y))
            painter.setBrush(QColor(DRUM_COLORS[lane]))
            painter.setPen(Qt.NoPen)
            for seconds in self.events.get(lane, []):
                if start <= seconds <= end:
                    x = rect.left() + label_width + (seconds - start) / self.window_seconds * (rect.width() - label_width - 8)
                    hit_active = abs(float(seconds) - self.position) <= active_window
                    radius = 15 if hit_active else 8
                    painter.drawEllipse(QRectF(x - radius, center_y - radius, radius * 2, radius * 2))

        painter.setPen(QPen(QColor("#ffffff"), 3))
        painter.drawLine(round(play_x), rect.top() + 4, round(play_x), rect.bottom() - 4)
        painter.setBrush(QColor("#ffffff"))
        painter.setPen(Qt.NoPen)
        painter.drawPolygon([
            QPointF(play_x, rect.top() + 3),
            QPointF(play_x - 6, rect.top() - 7),
            QPointF(play_x + 6, rect.top() - 7),
        ])

    def draw_beat_grid(self, painter, rect, label_width, start, end):
        width = rect.width() - label_width - 8
        downbeats = set(round(float(x), 4) for x in self.beat_grid.get("downbeats", []))
        for beat in self.beat_grid.get("beats", []):
            beat = float(beat)
            if start <= beat <= end:
                x = rect.left() + label_width + (beat - start) / self.window_seconds * width
                is_downbeat = round(beat, 4) in downbeats
                pen = QPen(QColor("#71658f") if is_downbeat else QColor("#3a3d49"))
                pen.setWidth(2 if is_downbeat else 1)
                painter.setPen(pen)
                painter.drawLine(round(x), rect.top() + 4, round(x), rect.bottom() - 4)


class BassRollView(QWidget):
    def __init__(self):
        super().__init__()
        self.notes = []
        self.beat_grid = {}
        self.position = 0.0
        self.window_seconds = 2.5
        self.setMinimumHeight(280)

    def set_events(self, events):
        self.notes = list(events.get("notes", []))
        self.update()

    def set_beat_grid(self, beat_grid):
        self.beat_grid = beat_grid or {}
        self.update()

    def set_position_ms(self, milliseconds):
        self.position = max(0.0, milliseconds / 1000)
        self.update()

    def pitch_y(self, pitch, rect):
        span = max(1, BASS_HIGH - BASS_LOW)
        ratio = (pitch - BASS_LOW) / span
        return rect.bottom() - ratio * rect.height()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(12, 12, -12, -12)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#1e212a"))
        painter.drawRoundedRect(rect, 8, 8)

        label_width = 56
        roll_rect = QRectF(rect.left() + label_width, rect.top() + 8, rect.width() - label_width - 8, rect.height() - 16)
        start = max(0.0, self.position)
        end = start + self.window_seconds
        play_x = roll_rect.left()

        painter.setPen(QPen(QColor("#30333d")))
        for pitch in range(BASS_LOW, BASS_HIGH + 1):
            if pitch % 12 == 0:
                y = self.pitch_y(pitch, roll_rect)
                painter.drawLine(round(roll_rect.left()), round(y), round(roll_rect.right()), round(y))
                painter.setPen(QPen(QColor("#89909f")))
                painter.drawText(QRectF(rect.left() + 8, y - 10, label_width - 14, 20), Qt.AlignRight | Qt.AlignVCenter, self.pitch_name(pitch))
                painter.setPen(QPen(QColor("#30333d")))

        downbeats = set(round(float(x), 4) for x in self.beat_grid.get("downbeats", []))
        for beat in self.beat_grid.get("beats", []):
            beat = float(beat)
            if start <= beat <= end:
                x = roll_rect.left() + (beat - start) / self.window_seconds * roll_rect.width()
                is_downbeat = round(beat, 4) in downbeats
                pen = QPen(QColor("#71658f") if is_downbeat else QColor("#3a3d49"))
                pen.setWidth(2 if is_downbeat else 1)
                painter.setPen(pen)
                painter.drawLine(round(x), round(roll_rect.top()), round(x), round(roll_rect.bottom()))

        painter.setPen(QPen(QColor("#aeb4c4")))
        painter.drawText(roll_rect.adjusted(8, 6, -8, -6), Qt.AlignLeft | Qt.AlignTop, f"Notes {len(self.notes)}")
        if not self.notes:
            painter.setPen(QPen(QColor("#7d8291")))
            painter.drawText(roll_rect, Qt.AlignCenter, getattr(self, "empty_text", "ベースMIDI未読込"))

        active_window = 0.08
        painter.setPen(Qt.NoPen)
        for note in self.notes:
            note_start = float(note.get("start", 0))
            note_end = float(note.get("end", note_start + 0.08))
            pitch = int(note.get("pitch", 40))
            if note_end < start or note_start > end or pitch < BASS_LOW or pitch > BASS_HIGH:
                continue
            x1 = roll_rect.left() + (max(note_start, start) - start) / self.window_seconds * roll_rect.width()
            x2 = roll_rect.left() + (min(note_end, end) - start) / self.window_seconds * roll_rect.width()
            y = self.pitch_y(pitch, roll_rect)
            active = note_start <= self.position <= note_end or abs(note_start - self.position) <= active_window
            height = 16 if active else 10
            painter.setBrush(QColor(getattr(self, "active_color", "#6ee7cf")) if active else QColor(getattr(self, "color", "#87d7c3")))
            painter.drawRoundedRect(QRectF(x1, y - height / 2, max(7.0, x2 - x1), height), 4, 4)

        painter.setPen(QPen(QColor("#ffffff"), 3))
        painter.drawLine(round(play_x), round(roll_rect.top()), round(play_x), round(roll_rect.bottom()))
        painter.setBrush(QColor("#ffffff"))
        painter.setPen(Qt.NoPen)
        painter.drawPolygon([
            QPointF(play_x, roll_rect.top() + 3),
            QPointF(play_x - 6, roll_rect.top() - 7),
            QPointF(play_x + 6, roll_rect.top() - 7),
        ])

    @staticmethod
    def pitch_name(pitch):
        names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
        return f"{names[pitch % 12]}{pitch // 12 - 1}"


class VocalRollView(BassRollView):
    low_pitch = VOCAL_LOW
    high_pitch = VOCAL_HIGH
    empty_text = "ボーカルMIDI未読込"
    color = "#c0abff"
    active_color = "#d8cffb"

    def pitch_y(self, pitch, rect):
        span = max(1, VOCAL_HIGH - VOCAL_LOW)
        ratio = (pitch - VOCAL_LOW) / span
        return rect.bottom() - ratio * rect.height()

    def paintEvent(self, event):
        old_low, old_high = globals()["BASS_LOW"], globals()["BASS_HIGH"]
        globals()["BASS_LOW"], globals()["BASS_HIGH"] = VOCAL_LOW, VOCAL_HIGH
        try:
            super().paintEvent(event)
        finally:
            globals()["BASS_LOW"], globals()["BASS_HIGH"] = old_low, old_high


ROOT_TO_PC = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}
PITCH_NAMES_UI = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")


def shape_to_text(shape):
    frets = shape.get("frets", []) if shape else []
    parts = []
    for fret in frets[:6]:
        if fret < 0:
            parts.append("x")
        else:
            parts.append(str(int(fret)))
    return " ".join(parts)


def parse_shape_text(value):
    raw = (value or "").strip()
    if not raw:
        raise ValueError("空のフォームです")
    if " " in raw or "," in raw:
        tokens = raw.replace(",", " ").split()
    else:
        tokens = list(raw)
    if len(tokens) != 6:
        raise ValueError("6弦分の指定が必要です。例: x 3 2 0 3 3")
    frets = []
    for token in tokens:
        t = token.strip().lower()
        if t in ("x", "-"):
            frets.append(-1)
        elif t in ("o", "open"):
            frets.append(0)
        else:
            frets.append(int(t))
    return frets


def load_user_chord_shapes():
    if not USER_CHORD_SHAPES_PATH.is_file():
        return {}
    try:
        data = json.loads(USER_CHORD_SHAPES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    result = {}
    for name, shape in data.items():
        frets = shape.get("frets") if isinstance(shape, dict) else None
        if isinstance(frets, list) and len(frets) == 6:
            result[name] = {"frets": [int(f) for f in frets], "barre": shape.get("barre")}
    return result


def save_user_chord_shapes(shapes):
    USER_CHORD_SHAPES_PATH.write_text(json.dumps(shapes, ensure_ascii=False, indent=2), encoding="utf-8")


USER_CHORD_SHAPES = load_user_chord_shapes()
SECONDARY_CHORD_SHAPES = json.loads((ROOT / "chord_shape_secondary.json").read_text(encoding="utf-8")) if (ROOT / "chord_shape_secondary.json").is_file() else {}

def save_secondary_chord_shapes():
    (ROOT / "chord_shape_secondary.json").write_text(
        json.dumps(SECONDARY_CHORD_SHAPES, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_chord_name(name):
    base = (name or "").replace("♭", "b").replace("♯", "#").strip()
    if not base or base == "N.C.":
        return None, "", None
    slash = None
    if "/" in base:
        base, slash = base.split("/", 1)
    root = base[:2] if len(base) >= 2 and base[1] in "#b" else base[:1]
    quality = base[len(root):]
    return root, quality, slash


QUALITY_ALIASES = {"maj": "", "M": "", "major": "", "min": "m", "minor": "m", "-": "m", "♭5": "b5", "ø": "m7b5", "ø7": "m7b5", "M7": "maj7", "△7": "maj7", "Δ7": "maj7"}


def normalize_quality(quality):
    q = (quality or "").replace(":", "").replace("♭", "b").replace("♯", "#").replace("△", "maj").replace("Δ", "maj")
    q = q.replace("−", "m").replace("-", "m")
    if q in QUALITY_ALIASES:
        return QUALITY_ALIASES[q]
    return q






def chord_edit_suggestions(name):
    root, quality, slash = parse_chord_name(name)
    if not root:
        return ["C", "Am", "G"]
    quality = normalize_quality(quality)
    # Keep the root and offer color tones from the same harmonic family.
    # A major triad must not suddenly suggest its parallel minor as the first
    # correction; borrowed/relative changes remain a manual-entry choice.
    families = {
        "": ["add9", "sus4", "6", "maj7", "7"],
        "m": ["m7", "m6", "madd9", "sus4", "m7b5"],
        "7": ["7sus4", "add9", "6", "maj7", "9"],
        "maj7": ["", "add9", "6", "maj9", "sus4"],
        "m7": ["m", "madd9", "m6", "sus4", "m9"],
        "m7b5": ["dim", "dim7", "m7", "m"],
        "dim": ["dim7", "m7b5", "m", "7"],
        "dim7": ["dim", "m7b5", "7"],
        "sus4": ["", "7sus4", "add9", "sus2", "7"],
        "sus2": ["", "add9", "sus4", "6"],
        "7sus4": ["sus4", "7", "add9", "9"],
        "add9": ["", "sus2", "sus4", "6", "maj7"],
        "madd9": ["m", "m7", "m6", "sus4"],
        "6": ["", "add9", "maj7", "7"],
        "m6": ["m", "m7", "madd9"],
    }
    alternatives = families.get(quality, ["", "add9", "sus4", "maj7", "7"])
    suffix = f"/{slash}" if slash else ""
    suggestions = []
    for q in alternatives:
        chord = root + q + suffix
        if chord != name and chord not in suggestions:
            suggestions.append(chord)
    # Also include enharmonic/current-root stable nearby basics if needed.
    for q in families.get(quality, ("", "add9", "sus4", "maj7", "7")):
        chord = root + q + suffix
        if chord != name and chord not in suggestions:
            suggestions.append(chord)
        if len(suggestions) >= 3:
            break
    return [candidate for candidate in suggestions if chord_shape(candidate) is not None][:5]


def chord_shape(name):
    # All fingerings come from chord_shape.json; never synthesize a fallback.
    if name in USER_CHORD_SHAPES:
        return USER_CHORD_SHAPES[name]
    root, quality, slash = parse_chord_name(name)
    if not root:
        return None
    quality = normalize_quality(quality)
    key = root + quality + (f"/{slash}" if slash else "")
    if key in USER_CHORD_SHAPES:
        return USER_CHORD_SHAPES[key]
    # Enharmonic spelling may reuse the same registered fingering.
    for registered, shape in USER_CHORD_SHAPES.items():
        other_root, other_quality, other_slash = parse_chord_name(registered)
        if (root in ROOT_TO_PC and other_root in ROOT_TO_PC and
                ROOT_TO_PC[root] == ROOT_TO_PC[other_root] and
                quality == normalize_quality(other_quality) and
                ((slash is None and other_slash is None) or
                 (slash in ROOT_TO_PC and other_slash in ROOT_TO_PC and
                  ROOT_TO_PC[slash] == ROOT_TO_PC[other_slash]))):
            return shape
    return None


class ChordDiagramButton(QPushButton):
    def __init__(self, name, shape=None):
        super().__init__()
        self.name = name
        self.shape = shape
        self.setMinimumSize(176, 152)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(name)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(4, 4, -4, -4)
        painter.setPen(QPen(QColor("#82739f") if self.isDown() else QColor("#3a3d49")))
        painter.setBrush(QColor("#30283f") if self.isDown() else QColor("#242731"))
        painter.drawRoundedRect(rect, 8, 8)
        painter.setPen(QPen(QColor("#f1c27d")))
        font = painter.font()
        font.setPointSize(14)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(rect.adjusted(8, 6, -8, -6), Qt.AlignHCenter | Qt.AlignTop, self.name)
        shape = self.shape or chord_shape(self.name)
        grid = QRectF(rect.left() + 12, rect.top() + 36, rect.width() - 24, rect.height() - 46)
        if shape:
            ChordTimelineView.draw_static_diagram(painter, grid, shape)
        else:
            font.setPointSize(10)
            painter.setFont(font)
            painter.drawText(grid, Qt.AlignCenter, "フォーム未登録")


class FretboardPickerDialog(QDialog):
    """Clickable six-string/fret picker used when the chord name is unknown."""
    OPEN_PITCHES = (4, 9, 2, 7, 11, 4)  # 6th string through 1st string
    QUALITIES = (("", (0, 4, 7)), ("m", (0, 3, 7)), ("7", (0, 4, 7, 10)),
                 ("maj7", (0, 4, 7, 11)), ("m7", (0, 3, 7, 10)),
                 ("m7b5", (0, 3, 6, 10)), ("sus4", (0, 5, 7)),
                 ("sus2", (0, 2, 7)), ("add9", (0, 2, 4, 7)), ("6", (0, 4, 7, 9)))

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("押さえ方からコードを指定")
        # Start with every string sounding open, which is the natural default
        # when entering a familiar guitar shape by clicking.
        self.frets = [0] * 6
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("弦を選び、フレットをクリックしてください（もう一度で次のフレット）"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("コード名を入力（例: Cadd9、Dm7、G/B）")
        self.name_input.textChanged.connect(self.apply_name_shape)
        layout.addWidget(self.name_input)
        grid = QGridLayout()
        grid.addWidget(QLabel("弦"), 0, 0)
        for fret in range(13):
            grid.addWidget(QLabel("ミュート" if fret == 0 else str(fret)), 0, fret + 1, alignment=Qt.AlignCenter)
        self.buttons = []
        for string in range(6):
            display_row = 6 - string
            grid.addWidget(QLabel(f"{6 - string}弦"), display_row, 0)
            row = []
            for fret in range(13):
                button = QPushButton("×" if fret == 0 else "")
                button.setFixedWidth(40)
                button.clicked.connect(lambda checked=False, s=string, f=fret: self.select_fret(s, f))
                grid.addWidget(button, display_row, fret + 1)
                row.append(button)
            self.buttons.append(row)
        layout.addLayout(grid)
        self.result = QLabel("初期状態は全弦開放です。必要な弦だけミュートまたはフレットを指定してください")
        self.result.setObjectName("muted")
        result_font = self.result.font()
        result_font.setPointSize(18)
        result_font.setBold(True)
        self.result.setFont(result_font)
        layout.addWidget(self.result)
        buttons = QHBoxLayout()
        cancel = QPushButton("キャンセル")
        cancel.clicked.connect(self.reject)
        accept = QPushButton("このフォームを使う")
        accept.clicked.connect(self.accept_shape)
        buttons.addWidget(cancel)
        secondary = QPushButton("セカンダリとして保存")
        secondary.clicked.connect(self.accept_secondary_shape)
        buttons.addWidget(secondary)
        buttons.addWidget(accept)
        layout.addLayout(buttons)

    def select_fret(self, string, fret):
        # The dedicated first column mutes the string. Other columns select
        # an explicit fret; an unmodified string remains open (0).
        if fret == 0:
            self.frets[string] = 0 if self.frets[string] < 0 else -1
        else:
            self.frets[string] = 0 if self.frets[string] == fret else fret
        for row in self.buttons:
            for button in row:
                button.setStyleSheet("")
        for string, fret in enumerate(self.frets):
            if fret < 0:
                self.buttons[string][0].setStyleSheet("background:#bea6ff;color:#20172f;")
            elif fret > 0:
                self.buttons[string][fret].setStyleSheet("background:#bea6ff;color:#20172f;")
        name = self.guess_name()
        self.result.setText(name or "コードを構成する音が2つ以上必要です")

    def set_shape(self, shape):
        if shape and len(shape.get("frets", [])) == 6:
            self.frets = list(shape["frets"])
        for string, fret in enumerate(self.frets):
            for candidate in self.buttons[string]:
                candidate.setStyleSheet("")
            if fret < 0:
                self.buttons[string][0].setStyleSheet("background:#bea6ff;color:#20172f;")
            elif fret > 0:
                self.buttons[string][fret].setStyleSheet("background:#bea6ff;color:#20172f;")
        name = self.guess_name()
        self.result.setText(name or "コードを構成する音が2つ以上必要です")

    def apply_name_shape(self, name):
        name = name.strip()
        if not name:
            return
        shape = chord_shape(name)
        if shape is not None:
            self.set_shape(shape)
            self.result.setText(f"{name} のフォーム")

    def guess_name(self):
        sounding = {(pitch + fret) % 12 for pitch, fret in zip(self.OPEN_PITCHES, self.frets) if fret >= 0}
        if len(sounding) < 2:
            return None
        # The first sounding string is the guitar's actual lowest note.  It
        # resolves pitch-set aliases such as x x 0 2 1 1: Dm7, not F6.
        lowest_pc = next(((pitch + fret) % 12 for pitch, fret in zip(self.OPEN_PITCHES, self.frets) if fret >= 0), None)
        best = None
        for root_name, root_pc in ROOT_TO_PC.items():
            if root_name not in PITCH_NAMES_UI:
                continue
            for quality, intervals in self.QUALITIES:
                pcs = {(root_pc + interval) % 12 for interval in intervals}
                score = len(sounding & pcs) / len(pcs) - 0.18 * len(sounding - pcs)
                score += 0.38 if root_pc == lowest_pc else (0.04 if lowest_pc in pcs else -0.12)
                if best is None or score > best[0]:
                    best = (score, root_name, quality)
        # Some useful open-string voicings (for example all strings open)
        # contain extensions outside the compact quality templates. They are
        # still valid multi-note guitar forms, so return the best musical
        # label instead of rejecting them solely on the template threshold.
        return best[1] + best[2] if best else None

    def accept_shape(self):
        name = self.guess_name()
        if not name:
            return
        USER_CHORD_SHAPES[name] = {"frets": list(self.frets), "barre": None}
        save_user_chord_shapes(USER_CHORD_SHAPES)
        self.selected = name
        self.accept()

    def accept_secondary_shape(self):
        name = self.name_input.text().strip() or self.guess_name()
        if not name:
            self.result.setText("コード名を入力するか、2音以上を選択してください")
            return
        SECONDARY_CHORD_SHAPES.setdefault(name, []).append({"frets": list(self.frets), "barre": None})
        save_secondary_chord_shapes()
        self.selected = name
        self.accept()


class ChordCandidateDialog(QDialog):
    def __init__(self, current, suggestions, parent=None):
        super().__init__(parent)
        self.selected = None
        self.selected_shape = None
        self.shape_touched = False
        self.frets = [0] * 6
        self.setWindowTitle("コードを修正")
        layout = QVBoxLayout(self)
        label = QLabel(f"{current} を修正")
        label.setObjectName("muted")
        layout.addWidget(label)
        grid = QGridLayout()
        for column, name in enumerate(suggestions):
            button = ChordDiagramButton(name)
            button.clicked.connect(lambda checked=False, value=name: self.select_candidate(value))
            grid.addWidget(button, 0, column)
        layout.addLayout(grid)
        layout.addWidget(QLabel("押さえ方を指定（初期状態は全弦開放）"))
        fret_grid = QGridLayout()
        fret_grid.addWidget(QLabel("弦"), 0, 0)
        for fret in range(13):
            fret_grid.addWidget(QLabel("ミュート" if fret == 0 else str(fret)), 0, fret + 1, alignment=Qt.AlignCenter)
        self.fret_buttons = []
        for string in range(6):
            display_row = 6 - string
            fret_grid.addWidget(QLabel(f"{6 - string}弦"), display_row, 0)
            row = []
            for fret in range(13):
                button = QPushButton("×" if fret == 0 else "")
                button.setFixedWidth(40)
                button.clicked.connect(lambda checked=False, s=string, f=fret: self.select_fret(s, f))
                fret_grid.addWidget(button, display_row, fret + 1)
                row.append(button)
            self.fret_buttons.append(row)
        layout.addLayout(fret_grid)
        manual_row = QHBoxLayout()
        manual_row.addWidget(QLabel("コード名"))
        self.manual = QLineEdit(current)
        self.manual.selectAll()
        self.manual.textChanged.connect(self.apply_name_shape)
        manual_row.addWidget(self.manual, 1)
        layout.addLayout(manual_row)
        self.result = QLabel()
        self.result.setObjectName("muted")
        layout.addWidget(self.result)
        actions = QHBoxLayout()
        actions.addStretch()
        cancel = QPushButton("キャンセル")
        cancel.clicked.connect(self.reject)
        actions.addWidget(cancel)
        accept = QPushButton("決定")
        accept.clicked.connect(self.choose_final)
        actions.addWidget(accept)
        layout.addLayout(actions)
        self.apply_name_shape(current)

    def refresh_frets(self):
        for string, fret in enumerate(self.frets):
            for button in self.fret_buttons[string]:
                button.setStyleSheet("")
            if fret < 0:
                self.fret_buttons[string][0].setStyleSheet("background:#bea6ff;color:#20172f;")
            elif fret > 0:
                self.fret_buttons[string][fret].setStyleSheet("background:#bea6ff;color:#20172f;")

    def apply_name_shape(self, name):
        shape = chord_shape(name.strip())
        if shape and len(shape.get("frets", [])) == 6:
            self.frets = list(shape["frets"])
            self.refresh_frets()
            self.result.setText(f"{name.strip()} のフォーム")

    def select_candidate(self, value):
        self.manual.setText(value)

    def guess_name(self):
        sounding = {(pitch + fret) % 12 for pitch, fret in zip(FretboardPickerDialog.OPEN_PITCHES, self.frets) if fret >= 0}
        if len(sounding) < 2:
            return None
        lowest_pc = next(((pitch + fret) % 12 for pitch, fret in zip(FretboardPickerDialog.OPEN_PITCHES, self.frets) if fret >= 0), None)
        best = None
        for root_name, root_pc in ROOT_TO_PC.items():
            if root_name not in PITCH_NAMES_UI:
                continue
            for quality, intervals in FretboardPickerDialog.QUALITIES:
                pcs = {(root_pc + interval) % 12 for interval in intervals}
                score = len(sounding & pcs) / len(pcs) - .18 * len(sounding - pcs)
                score += .38 if root_pc == lowest_pc else (.04 if lowest_pc in pcs else -.12)
                if best is None or score > best[0]:
                    best = (score, root_name, quality)
        return best[1] + best[2] if best else None

    def select_fret(self, string, fret):
        self.frets[string] = (0 if self.frets[string] < 0 else -1) if fret == 0 else (0 if self.frets[string] == fret else fret)
        self.shape_touched = True
        self.refresh_frets()
        guessed = self.guess_name()
        # Reflect the newly specified fingering in the editable code field,
        # without applying a registered form back over the user's clicks.
        self.manual.blockSignals(True)
        self.manual.setText(guessed or "")
        self.manual.blockSignals(False)
        self.result.setText(guessed or "押さえ方を指定してください")

    def choose_final(self):
        value = self.manual.text().strip() or self.guess_name()
        if not value:
            self.result.setText("コード名を入力するか、押さえ方を指定してください")
            return
        # A timeline correction is local to this chord occurrence. Global
        # dictionary updates belong exclusively to the code-table editor.
        self.selected_shape = {"frets": list(self.frets), "barre": None} if self.shape_touched else None
        self.selected = value
        self.accept()


class ChordTimelineView(QWidget):
    chordEdited = Signal(int, str, float, float)

    def __init__(self):
        super().__init__()
        self.chords = []
        self.position = 0.0
        self.window_seconds = 10.0
        self.card_rects = []
        self.editable = True
        self.downbeats = []
        self.beats = []
        self.lane = QRectF()
        self.hover_time = None
        self.drag_anchor_time = None
        self.drag_time = None
        self.drag_index = None
        self.setMinimumHeight(190)
        self.setMouseTracking(True)
        self.setToolTip("クリックまたはドラッグで四分音符を選択してコードを修正")

    def set_chords(self, chords):
        self.chords = list((chords or {}).get("chords", []))
        self.update()

    def set_beat_grid(self, grid):
        self.beats = sorted({float(t) for t in (grid or {}).get("beats", [])
                             if np.isfinite(float(t)) and float(t) >= 0})
        self.downbeats = sorted({float(t) for t in (grid or {}).get("downbeats", [])
                                 if np.isfinite(float(t)) and float(t) >= 0})
        self.update()

    def set_position_ms(self, milliseconds):
        self.position = max(0.0, milliseconds / 1000)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(12, 12, -12, -12)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#1e212a"))
        painter.drawRoundedRect(rect, 8, 8)
        play_x = rect.left() + 18
        lane = QRectF(play_x, rect.top() + 8, rect.width() - 26, rect.height() - 16)
        self.lane = lane
        start = self.position
        end = start + self.window_seconds
        self.card_rects = []
        painter.setPen(QPen(QColor("#aeb4c4")))
        hint = "クリック/ドラッグした四分音符を修正" if self.editable else "Solitito · アコギ"
        painter.drawText(lane.adjusted(8, 6, -8, -6), Qt.AlignLeft | Qt.AlignTop, f"Chords {len(self.chords)}  ·  {hint}")
        if not self.chords:
            painter.setPen(QPen(QColor("#7d8291")))
            painter.drawText(lane, Qt.AlignCenter, "コード未推定")
        painter.save()
        painter.setClipRect(lane)
        for index, item in enumerate(self.chords):
            c_start = float(item.get("start", 0))
            c_end = float(item.get("end", c_start + 1))
            if c_end <= start or c_start >= end:
                continue
            x = lane.left() + (c_start - start) / self.window_seconds * lane.width()
            width = max(86.0, (c_end - c_start) / self.window_seconds * lane.width())
            card = QRectF(x, lane.top() + 28, width, lane.height() - 34)
            visible = QRectF(x, card.top(), (c_end - c_start) / self.window_seconds * lane.width(), card.height()).intersected(lane)
            self.card_rects.append((visible, index))
            painter.save()
            painter.setClipRect(visible, Qt.IntersectClip)
            self.draw_chord_card(painter, card, item.get("chord", "N.C."), float(item.get("confidence", 0)), item.get("shape"))
            painter.restore()
        selected = self.selected_quarter_range() or self.quarter_cell_at(self.hover_time)
        if selected:
            cell_start, cell_end = selected
            left = lane.left() + (cell_start - start) / self.window_seconds * lane.width()
            right = lane.left() + (cell_end - start) / self.window_seconds * lane.width()
            highlight = QRectF(left, lane.top() + 26, right - left, lane.height() - 26).intersected(lane)
            painter.setPen(QPen(QColor("#f1c27d"), 2))
            painter.setBrush(QColor(241, 194, 125, 42))
            painter.drawRect(highlight)
        # Draw over the cards so sustained chords do not conceal bar boundaries.
        painter.setPen(QPen(QColor("#969fb7"), 1, Qt.DashLine))
        for downbeat in self.downbeats:
            if start <= downbeat <= end:
                x = lane.left() + (downbeat - start) / self.window_seconds * lane.width()
                painter.drawLine(QPointF(x, lane.top() + 25), QPointF(x, lane.bottom()))
        painter.restore()
        painter.setPen(QPen(QColor("#ffffff"), 3))
        painter.drawLine(round(play_x), rect.top() + 4, round(play_x), rect.bottom() - 4)
        painter.setBrush(QColor("#ffffff"))
        painter.setPen(Qt.NoPen)
        painter.drawPolygon([QPointF(play_x, rect.top() + 3), QPointF(play_x - 6, rect.top() - 7), QPointF(play_x + 6, rect.top() - 7)])


    def mousePressEvent(self, event):
        if not self.editable:
            return super().mousePressEvent(event)
        for card, index in reversed(self.card_rects):
            if card.contains(event.position()):
                cursor_time = self.position + (event.position().x() - self.lane.left()) / self.lane.width() * self.window_seconds
                self.drag_anchor_time = max(0.0, cursor_time)
                self.drag_time = self.drag_anchor_time
                self.drag_index = index
                self.update()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.editable and self.lane.contains(event.position()):
            cursor_time = self.position + (event.position().x() - self.lane.left()) / self.lane.width() * self.window_seconds
            self.hover_time = cursor_time
            if self.drag_anchor_time is not None:
                self.drag_time = cursor_time
        else:
            self.hover_time = None
        self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.drag_index is not None:
            selected = self.selected_quarter_range()
            index = self.drag_index
            self.drag_anchor_time = self.drag_time = self.drag_index = None
            if selected:
                self.chordEdited.emit(index, self.chords[index].get("chord", ""), *selected)
            self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        self.hover_time = None
        self.update()
        super().leaveEvent(event)

    def quarter_cell_at(self, time):
        """Find the actual beat-grid quarter-note cell below a cursor time."""
        if time is None:
            return None
        for cell_start, cell_end in zip(self.beats, self.beats[1:]):
            if cell_start <= time < cell_end:
                return cell_start, cell_end
        return None

    def selected_quarter_range(self):
        """Snap a drag selection outwards to complete quarter-note cells."""
        first = self.quarter_cell_at(self.drag_anchor_time)
        last = self.quarter_cell_at(self.drag_time)
        if not first or not last:
            return None
        return min(first[0], last[0]), max(first[1], last[1])

    def draw_chord_card(self, painter, card, name, confidence, local_shape=None):
        painter.setPen(QPen(QColor("#3a3d49")))
        painter.setBrush(QColor("#242731"))
        painter.drawRoundedRect(card, 8, 8)
        painter.setPen(QPen(QColor("#f1c27d")))
        font = painter.font()
        font.setPointSize(13)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(card.adjusted(8, 4, -8, -4), Qt.AlignLeft | Qt.AlignTop, name)
        font.setPointSize(8)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QPen(QColor("#8f93a3")))
        painter.drawText(card.adjusted(8, 22, -8, -4), Qt.AlignLeft | Qt.AlignTop, f"{confidence:.2f}")
        if not self.editable and parse_chord_name(name)[1] == "sus":
            painter.drawText(card.adjusted(8, 48, -8, -8), Qt.AlignCenter, "sus2 / sus4\n未判定")
            return
        shape = local_shape or chord_shape(name)
        if not shape or all(f < 0 for f in shape.get("frets", [])):
            painter.setPen(QPen(QColor("#8f93a3")))
            painter.drawText(card.adjusted(8, 48, -8, -8), Qt.AlignCenter, "フォーム未登録" if name != "N.C." else name)
            return
        grid = QRectF(card.left() + 10, card.top() + 46, min(86, card.width() - 18), card.height() - 56)
        self.draw_diagram(painter, grid, shape)

    @staticmethod
    def draw_static_diagram(painter, grid, shape):
        frets = shape["frets"] if shape else [-1, -1, -1, -1, -1, -1]
        sounding = [f for f in frets if f > 0]
        base = max(1, min(sounding) if sounding else 1)
        if base <= 3:
            base = 1
        fret_count = 5
        left = grid.left() + 14
        right = grid.right() - 6
        top = grid.top() + 4
        bottom = grid.bottom() - 12
        string_gap = (bottom - top) / 5
        fret_gap = (right - left) / fret_count
        painter.setPen(QPen(QColor("#c4c7d4"), 1))
        for i in range(6):
            y = bottom - i * string_gap
            painter.drawLine(round(left), round(y), round(right), round(y))
        for i in range(fret_count + 1):
            x = left + i * fret_gap
            pen = QPen(QColor("#c4c7d4"), 3 if i == 0 and base == 1 else 1)
            painter.setPen(pen)
            painter.drawLine(round(x), round(top), round(x), round(bottom))
        painter.setPen(QPen(QColor("#ffffff")))
        font = painter.font(); font.setPointSize(9); font.setBold(True); painter.setFont(font)
        for i in range(fret_count):
            painter.drawText(QRectF(left + i * fret_gap, bottom + 1, fret_gap, 15), Qt.AlignCenter, str(base + i))
        barre = shape.get("barre") if shape else None
        if barre:
            fret, string_from, string_to = barre
            if base <= fret < base + fret_count:
                x = left + (fret - base + 0.5) * fret_gap
                y1 = bottom - string_from * string_gap
                y2 = bottom - string_to * string_gap
                painter.setPen(QPen(QColor("#f1c27d"), 9, Qt.SolidLine, Qt.RoundCap))
                painter.drawLine(round(x), round(y1), round(x), round(y2))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#f1c27d"))
        for string_index, fret in enumerate(frets):
            y = bottom - string_index * string_gap
            if fret < 0:
                painter.setPen(QPen(QColor("#8f93a3")))
                painter.drawText(QRectF(left - 14, y - 7, 12, 14), Qt.AlignCenter, "x")
                painter.setPen(Qt.NoPen)
            elif fret == 0:
                painter.setPen(QPen(QColor("#8f93a3")))
                painter.drawText(QRectF(left - 14, y - 7, 12, 14), Qt.AlignCenter, "o")
                painter.setPen(Qt.NoPen)
            elif base <= fret < base + fret_count:
                x = left + (fret - base + 0.5) * fret_gap
                painter.drawEllipse(QRectF(x - 5, y - 5, 10, 10))

    def draw_diagram(self, painter, grid, shape):
        self.draw_static_diagram(painter, grid, shape)
        return
        frets = shape["frets"]
        sounding = [f for f in frets if f > 0]
        base = max(1, min(sounding) if sounding else 1)
        if base <= 3:
            base = 1
        fret_count = 5
        left = grid.left() + 14
        right = grid.right() - 6
        top = grid.top() + 4
        bottom = grid.bottom() - 8
        string_gap = (bottom - top) / 5
        fret_gap = (right - left) / fret_count
        painter.setPen(QPen(QColor("#c4c7d4"), 1))
        for i in range(6):
            y = bottom - i * string_gap  # 6th string is bottom.
            painter.drawLine(round(left), round(y), round(right), round(y))
        for i in range(fret_count + 1):
            x = left + i * fret_gap
            pen = QPen(QColor("#c4c7d4"), 3 if i == 0 and base == 1 else 1)
            painter.setPen(pen)
            painter.drawLine(round(x), round(top), round(x), round(bottom))
        painter.setPen(QPen(QColor("#ffffff")))
        font = painter.font(); font.setPointSize(9); font.setBold(True); painter.setFont(font)
        for i in range(fret_count):
            painter.drawText(QRectF(left + i * fret_gap, bottom + 1, fret_gap, 15), Qt.AlignCenter, str(base + i))
        barre = shape.get("barre")
        if barre:
            fret, string_from, string_to = barre
            if base <= fret < base + fret_count:
                x = left + (fret - base + 0.5) * fret_gap
                y1 = bottom - string_from * string_gap
                y2 = bottom - string_to * string_gap
                painter.setPen(QPen(QColor("#f1c27d"), 9, Qt.SolidLine, Qt.RoundCap))
                painter.drawLine(round(x), round(y1), round(x), round(y2))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#f1c27d"))
        for string_index, fret in enumerate(frets):
            y = bottom - string_index * string_gap
            if fret < 0:
                painter.setPen(QPen(QColor("#8f93a3")))
                painter.drawText(QRectF(left - 14, y - 7, 12, 14), Qt.AlignCenter, "x")
                painter.setPen(Qt.NoPen)
            elif fret == 0:
                painter.setPen(QPen(QColor("#8f93a3")))
                painter.drawText(QRectF(left - 14, y - 7, 12, 14), Qt.AlignCenter, "o")
                painter.setPen(Qt.NoPen)
            elif base <= fret < base + fret_count:
                x = left + (fret - base + 0.5) * fret_gap
                painter.drawEllipse(QRectF(x - 5, y - 5, 10, 10))


class PianoRollView(QWidget):
    def __init__(self):
        super().__init__()
        self.notes = []
        self.beat_grid = {}
        self.position = 0.0
        self.window_seconds = 2.5
        self.setMinimumHeight(520)

    def set_events(self, events):
        self.notes = list(events.get("notes", []))
        self.update()

    def set_beat_grid(self, beat_grid):
        self.beat_grid = beat_grid or {}
        self.update()

    def set_position_ms(self, milliseconds):
        self.position = max(0.0, milliseconds / 1000)
        self.update()

    @staticmethod
    def is_black_key(pitch):
        return pitch % 12 in {1, 3, 6, 8, 10}

    def white_pitches(self):
        return [pitch for pitch in range(PIANO_LOW, PIANO_HIGH + 1) if not self.is_black_key(pitch)]

    def pitch_x(self, pitch, keyboard_rect):
        white_pitches = self.white_pitches()
        white_index = {pitch: i for i, pitch in enumerate(white_pitches)}
        white_width = keyboard_rect.width() / len(white_pitches)
        if not self.is_black_key(pitch):
            return keyboard_rect.left() + (white_index[pitch] + 0.5) * white_width
        previous_white = pitch - 1
        while previous_white >= PIANO_LOW and self.is_black_key(previous_white):
            previous_white -= 1
        if previous_white in white_index:
            return keyboard_rect.left() + (white_index[previous_white] + 1.0) * white_width
        return keyboard_rect.left()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(12, 12, -12, -12)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#1e212a"))
        painter.drawRoundedRect(rect, 8, 8)

        keyboard_height = 72
        roll_rect = QRectF(rect.left() + 10, rect.top() + 8, rect.width() - 20, rect.height() - keyboard_height - 18)
        keyboard_rect = QRectF(roll_rect.left(), roll_rect.bottom() + 4, roll_rect.width(), keyboard_height)
        start = max(0.0, self.position)
        end = start + self.window_seconds
        white_pitches = self.white_pitches()
        white_width = keyboard_rect.width() / len(white_pitches)
        active_pitches = {
            int(note.get("pitch", -1))
            for note in self.notes
            if float(note.get("start", 0)) <= self.position <= float(note.get("end", 0))
        }

        painter.setPen(QPen(QColor("#30333d")))
        for pitch in range(PIANO_LOW, PIANO_HIGH + 1):
            if pitch % 12 == 0:
                x = self.pitch_x(pitch, roll_rect)
                painter.drawLine(round(x), round(roll_rect.top()), round(x), round(roll_rect.bottom()))

        downbeats = set(round(float(x), 4) for x in self.beat_grid.get("downbeats", []))
        for beat in self.beat_grid.get("beats", []):
            beat = float(beat)
            if start <= beat <= end:
                y = roll_rect.bottom() - (beat - start) / self.window_seconds * roll_rect.height()
                is_downbeat = round(beat, 4) in downbeats
                pen = QPen(QColor("#71658f") if is_downbeat else QColor("#3a3d49"))
                pen.setWidth(2 if is_downbeat else 1)
                painter.setPen(pen)
                painter.drawLine(round(roll_rect.left()), round(y), round(roll_rect.right()), round(y))

        painter.setPen(QPen(QColor("#aeb4c4")))
        painter.drawText(roll_rect.adjusted(8, 6, -8, -6), Qt.AlignLeft | Qt.AlignTop, f"Notes {len(self.notes)}")
        if not self.notes:
            painter.setPen(QPen(QColor("#7d8291")))
            painter.drawText(roll_rect, Qt.AlignCenter, "ピアノMIDI未読込")

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#9ed8ff"))
        note_width = max(7.0, white_width * 0.75)
        for note in self.notes:
            note_start = float(note.get("start", 0))
            note_end = float(note.get("end", note_start + 0.08))
            pitch = int(note.get("pitch", 60))
            if note_end < start or note_start > end or pitch < PIANO_LOW or pitch > PIANO_HIGH:
                continue
            x = self.pitch_x(pitch, roll_rect)
            y1 = roll_rect.bottom() - (max(note_start, start) - start) / self.window_seconds * roll_rect.height()
            y2 = roll_rect.bottom() - (min(note_end, end) - start) / self.window_seconds * roll_rect.height()
            note_top = min(y1, y2)
            note_bottom = max(y1, y2)
            painter.drawRoundedRect(QRectF(x - note_width / 2, note_top, note_width, max(4.0, note_bottom - note_top)), 2, 2)

        for index, pitch in enumerate(white_pitches):
            x = keyboard_rect.left() + index * white_width
            active = pitch in active_pitches
            painter.setBrush(QColor("#68d6ff") if active else QColor("#ffffff"))
            painter.setPen(QPen(QColor("#777b87")))
            painter.drawRect(QRectF(x, keyboard_rect.top(), white_width, keyboard_rect.height()))

        black_width = white_width * 0.62
        black_height = keyboard_rect.height() * 0.60
        for pitch in range(PIANO_LOW, PIANO_HIGH + 1):
            if not self.is_black_key(pitch):
                continue
            x = self.pitch_x(pitch, keyboard_rect)
            active = pitch in active_pitches
            painter.setBrush(QColor("#68d6ff") if active else QColor("#111217"))
            painter.setPen(QPen(QColor("#2c2f38")))
            painter.drawRoundedRect(QRectF(x - black_width / 2, keyboard_rect.top(), black_width, black_height), 2, 2)

        painter.setPen(QPen(QColor("#ffffff"), 3))
        painter.drawLine(round(roll_rect.left()), round(roll_rect.bottom()), round(roll_rect.right()), round(roll_rect.bottom()))


class Worker(QThread):
    log = Signal(str)
    progress = Signal(int)
    result = Signal(object)
    failed = Signal(str)

    def __init__(self, sources, output, count, output_format="int24"):
        super().__init__()
        self.sources, self.output, self.count = list(sources), output, count
        self.output_format = output_format
        self.runner = Separation()

    def run(self):
        try:
            last_folder = None
            total = len(self.sources)
            for index, source in enumerate(self.sources, 1):
                if self.runner.cancelled.is_set():
                    self.result.emit(None)
                    return
                self.log.emit(f"Batch {index}/{total}: {Path(source).name}")

                def batch_progress(value, offset=index - 1):
                    self.progress.emit(round(((offset * 100) + value) / total))

                last_folder = self.runner.run(source, self.output, self.count,
                                              self.log.emit, batch_progress, self.output_format)
                if last_folder is None:
                    self.result.emit(None)
                    return
            self.result.emit(last_folder)
        except Exception as error:
            self.failed.emit(str(error))


class DrumMidiWorker(QThread):
    result = Signal(object)
    failed = Signal(str)

    def __init__(self, folder, force=False):
        super().__init__()
        self.folder = Path(folder)
        self.force = force

    def run(self):
        try:
            self.result.emit(load_or_estimate(self.folder, force=self.force))
        except Exception as error:
            self.failed.emit(str(error))


class PianoMidiWorker(QThread):
    result = Signal(object)
    failed = Signal(str)

    def __init__(self, folder, force=False):
        super().__init__()
        self.folder = Path(folder)
        self.force = force

    def run(self):
        try:
            self.result.emit(load_or_estimate_piano(self.folder, force=self.force))
        except Exception as error:
            self.failed.emit(str(error))


class VocalMidiWorker(QThread):
    result = Signal(object)
    failed = Signal(str)

    def __init__(self, folder, force=False):
        super().__init__()
        self.folder = Path(folder)
        self.force = force

    def run(self):
        try:
            self.result.emit(load_or_estimate_vocal(self.folder, force=self.force))
        except Exception as error:
            self.failed.emit(str(error))


class BassMidiWorker(QThread):
    result = Signal(object)
    failed = Signal(str)

    def __init__(self, folder, force=False):
        super().__init__()
        self.folder = Path(folder)
        self.force = force

    def run(self):
        try:
            self.result.emit(load_or_estimate_bass(self.folder, force=self.force))
        except Exception as error:
            self.failed.emit(str(error))


class BeatGridWorker(QThread):
    result = Signal(object)
    failed = Signal(str)

    def __init__(self, folder, force=False):
        super().__init__()
        self.folder = Path(folder)
        self.force = force

    def run(self):
        try:
            self.result.emit(load_or_estimate_beats(self.folder, force=self.force))
        except Exception as error:
            self.failed.emit(str(error))


class FullAnalysisWorker(QThread):
    status = Signal(str)
    result = Signal(object)
    failed = Signal(str)

    def __init__(self, folder, force=False):
        super().__init__()
        self.folder = Path(folder)
        self.force = force

    def run(self):
        try:
            result = {}
            self.status.emit("BPM/拍グリッドを推定中…")
            result["beat_grid"] = load_or_estimate_beats(self.folder, force=self.force)
            if (self.folder / "vocals.wav").is_file():
                self.status.emit("ボーカルMIDIを推定中…")
                result["vocals"] = load_or_estimate_vocal(self.folder, force=self.force)
            if (self.folder / "drums.wav").is_file():
                self.status.emit("ドラムMIDIを推定中…")
                result["drums"] = load_or_estimate(self.folder, force=self.force)
            if (self.folder / "piano.wav").is_file():
                self.status.emit("ピアノMIDIを推定中…")
                result["piano"] = load_or_estimate_piano(self.folder, force=self.force)
            if (self.folder / "bass.wav").is_file():
                self.status.emit("ベースMIDIを推定中…")
                result["bass"] = load_or_estimate_bass(self.folder, force=self.force)
            self.result.emit(result)
        except Exception as error:
            self.failed.emit(str(error))


class ChordAnalysisWorker(QThread):
    """Recompute only the integrated chord result on every request."""
    status = Signal(str)
    result = Signal(object)
    failed = Signal(str)

    def __init__(self, folder):
        super().__init__()
        self.folder = Path(folder)

    def run(self):
        try:
            self.status.emit("アコギのコードを推定中…")
            acoustic = load_or_estimate_acoustic(self.folder, force=True)
            self.status.emit("アコギを基準にコードを再推定中…")
            drums_path = self.folder / "drums-midi.json"
            drums = json.loads(drums_path.read_text(encoding="utf-8")) if drums_path.is_file() else {}
            chords = load_or_estimate_chords(self.folder, force=True, acoustic=acoustic, drums=drums)
            self.result.emit({"acoustic": acoustic, "chords": chords})
        except Exception as error:
            self.failed.emit(str(error))


class DropZone(QPushButton):
    picked = Signal(object)

    def __init__(self):
        super().__init__("＋\n音源をドラッグ＆ドロップ\nまたは、クリックしてファイルを選択")
        self.setObjectName("drop")
        self.setMinimumHeight(164)
        self.setAcceptDrops(True)
        self.clicked.connect(self.browse)

    def browse(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "音源を選択", "",
                    "Audio (" + " ".join("*" + ext for ext in sorted(EXTENSIONS)) + ")")
        if paths:
            self.picked.emit(paths)

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls()
        if urls and all(url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() in EXTENSIONS for url in urls):
            event.acceptProposedAction()
            self.setProperty("dragging", True)
            self.style().unpolish(self)
            self.style().polish(self)

    def dragLeaveEvent(self, event):
        self.setProperty("dragging", False)
        self.style().unpolish(self)
        self.style().polish(self)

    def dropEvent(self, event):
        self.dragLeaveEvent(event)
        self.picked.emit([url.toLocalFile() for url in event.mimeData().urls()])
        event.acceptProposedAction()


class AcousticChordView(ChordTimelineView):
    """Solitito chords using the shared scrolling diagram timeline."""
    def __init__(self):
        super().__init__()
        self.editable = True
        self.setToolTip("アコギの推定コード · クリック/ドラッグした四分音符を修正 · 小節線は拍グリッドに同期")

    def current_chord(self):
        return next((c["chord"] for c in self.chords
                     if c["start"] <= self.position < c["end"]), "—")


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("STEM Studio")
        self.resize(820, 860)
        self.setMinimumSize(650, 650)
        self.sources = []
        self.output = ROOT / "outputs"
        self.result_folder = None
        self.vocal_view = None
        self.vocal_view = None
        self.drum_view = None
        self.piano_view = None
        self.bass_view = None
        self.chord_view = None
        self.acoustic_chord_view = None
        self.acoustic_chord_views = []
        self.acoustic_process = None
        self.acoustic_chords = {}
        self.vocal_note = None
        self.vocal_note = None
        self.drum_note = None
        self.piano_note = None
        self.bass_note = None
        self.beat_grid = {}
        self.worker = None
        self.vocal_worker = None
        self.drum_worker = None
        self.piano_worker = None
        self.bass_worker = None
        self.beat_worker = None
        self.full_analysis_worker = None
        self.play_buttons = {}
        self.seek_controls = {}
        self.pending_seek = None
        self.smooth_position = 0
        self.smooth_anchor = 0
        self.drum_visual_offset_ms = 0
        self.chords = {}
        self.smooth_clock = QElapsedTimer()
        self.audio = QAudioOutput(self)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio)
        self.audio.setVolume(0.8)
        self.player.playbackStateChanged.connect(self.sync_play_buttons)
        self.player.positionChanged.connect(self.update_play_position)
        self.player.mediaStatusChanged.connect(self.media_ready)
        self.player.errorOccurred.connect(lambda *_: self.status.setText("試聴エラー: " + self.player.errorString()))
        self.playback_timer = QTimer(self)
        self.playback_timer.setTimerType(Qt.PreciseTimer)
        self.playback_timer.setInterval(12)
        self.playback_timer.timeout.connect(self.refresh_playback_position)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.setCentralWidget(scroll)
        panel = QWidget()
        scroll.setWidget(panel)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(36, 28, 36, 28)
        layout.setSpacing(18)
        title = QLabel("STEM Studio - Music Separation Tool")
        title.setObjectName("title")
        layout.addWidget(title)
        
        self.drop = DropZone()
        self.drop.picked.connect(self.select_sources)
        layout.addWidget(self.drop)
        self.file_info = QLabel("")
        self.file_info.setWordWrap(True)
        self.file_info.setObjectName("muted")
        layout.addWidget(self.file_info)

        destination_row = QHBoxLayout()
        self.destination = QLabel()
        self.destination.setWordWrap(True)
        self.update_destination()
        destination_row.addWidget(self.destination, 1)
        self.browse_output = QPushButton("保存先を変更")
        self.browse_output.clicked.connect(self.choose_output)
        destination_row.addWidget(self.browse_output)
        self.load_existing = QPushButton("処理済みフォルダを開く")
        self.load_existing.clicked.connect(self.choose_existing_result)
        destination_row.addWidget(self.load_existing)
        layout.addLayout(destination_row)
        specialist_row = QHBoxLayout()
        self.specialist_model = QComboBox()
        self.specialist_model.setAccessibleName("パート別モデル")
        for key, (label, _, _) in SPECIALISTS.items():
            self.specialist_model.addItem(label, key)
        self.specialist_model.setCurrentIndex(self.specialist_model.findData("bs_sw_acoustic_guitar"))
        specialist_row.addWidget(self.specialist_model, 1)
        self.specialist_start = QPushButton("分離する →")
        self.specialist_start.setObjectName("primary")
        self.specialist_start.setMinimumHeight(44)
        self.specialist_start.setEnabled(False)
        self.specialist_start.clicked.connect(self.separate)
        specialist_row.addWidget(self.specialist_start)
        layout.addLayout(specialist_row)
        self.cancel = QPushButton("キャンセル")
        self.cancel.setVisible(False)
        self.cancel.clicked.connect(self.cancel_job)
        layout.addWidget(self.cancel, alignment=Qt.AlignRight)
        self.status = QLabel("音源を選択すると分離できます。")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(10)
        progress_row = QHBoxLayout()
        progress_row.addWidget(self.progress, 1)
        self.progress_text = QLabel("0%")
        self.progress_text.setObjectName("progressText")
        self.progress_text.setFixedWidth(64)
        self.progress_text.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        progress_row.addWidget(self.progress_text)
        layout.addLayout(progress_row)
        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("再生速度"))
        self.speed_combo = QComboBox()
        for label, rate in (("0.5x", 0.5), ("0.75x", 0.75), ("1.0x", 1.0), ("1.25x", 1.25), ("1.5x", 1.5)):
            self.speed_combo.addItem(label, rate)
        self.speed_combo.setCurrentIndex(2)
        self.speed_combo.currentIndexChanged.connect(self.change_playback_rate)
        speed_row.addWidget(self.speed_combo)
        self.analyze_prereq = QPushButton("BPM/MIDI推定")
        self.analyze_prereq.clicked.connect(lambda checked=False: self.load_full_analysis(force=False))
        self.analyze_prereq.setEnabled(False)
        speed_row.addWidget(self.analyze_prereq)
        self.analyze_chords = QPushButton("コード解析")
        self.analyze_chords.clicked.connect(self.load_chord_analysis)
        self.analyze_chords.setEnabled(False)
        speed_row.addWidget(self.analyze_chords)
        self.bpm_label = QLabel("BPM: 未推定")
        self.bpm_label.setObjectName("muted")
        self.bpm_label.setWordWrap(True)
        speed_row.addWidget(self.bpm_label, 1)
        layout.addLayout(speed_row)

        self.open_output = QPushButton("分離したファイルをFinderで開く ↗")
        self.open_output.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.result_folder))))
        self.open_output.hide()
        self.export_video = QPushButton("MIDIビュー動画を書き出す")
        self.export_video.clicked.connect(self.export_audio_video)
        self.export_video.setEnabled(False)
        output_actions = QHBoxLayout()
        output_actions.addWidget(self.open_output, 1)
        output_actions.addWidget(self.export_video, 1)
        layout.addLayout(output_actions)

        self.results = QVBoxLayout()
        layout.addLayout(self.results)
        self.logs = QPlainTextEdit()
        self.logs.setReadOnly(True)
        self.logs.setMaximumBlockCount(500)
        self.logs.setFixedHeight(135)
        self.logs.hide()
        layout.addWidget(self.logs)
        self.chord_table_panel = QWidget()
        self.chord_table_loaded = True
        self.chord_table_panel.setMinimumHeight(912)
        table_layout = QVBoxLayout(self.chord_table_panel)
        table_layout.setContentsMargins(0, 8, 0, 0)
        self.chord_table = QTableWidget()
        self.chord_table.setColumnCount(1)
        self.chord_table.horizontalHeader().hide()
        self.chord_table.verticalHeader().hide()
        self.chord_table.setMinimumHeight(672)
        table_layout.addWidget(self.chord_table)
        layout.addWidget(self.chord_table_panel)
        self.populate_chord_table()
        chord_tools = QHBoxLayout()
        chord_tools.addStretch()
        self.fingering_search_button = QPushButton("押さえ方から検索")
        self.fingering_search_button.clicked.connect(self.search_chord_by_fingering)
        chord_tools.addWidget(self.fingering_search_button)
        self.chord_table_button = QPushButton("コード表")
        self.chord_table_button.clicked.connect(self.toggle_chord_table)
        chord_tools.addWidget(self.chord_table_button)
        layout.addLayout(chord_tools)
        footer = QLabel("LOCAL PROCESSING  ·  音源は外部に送信されません  ·  WAV出力")
        footer.setObjectName("muted")
        footer.setWordWrap(True)
        layout.addWidget(footer)
        layout.addStretch()
        # Keep the debug loop quick: open the standard processed example when
        # it is available, while still starting empty on a fresh checkout.
        QTimer.singleShot(0, self.open_default_result)

    def chord_table_names(self):
        roots = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
        qualities = ["", "m", "7", "maj7", "m7", "m7b5", "dim", "dim7", "sus2", "sus4", "7sus4", "add9", "madd9", "9", "maj9", "m9", "11", "m11", "13", "m13", "6", "m6", "6/9"]
        names = {root + quality for root in roots for quality in qualities}
        names.update(USER_CHORD_SHAPES.keys())
        return sorted(names, key=lambda name: (ROOT_TO_PC.get(parse_chord_name(name)[0], 99), normalize_quality(parse_chord_name(name)[1]), name))

    def populate_chord_table(self):
        if not hasattr(self, "chord_table"):
            return
        self.chord_table.blockSignals(True)
        roots = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
        base_qualities = ["", "m", "7", "maj7", "m7", "sus4", "7sus4", "add9", "6", "9", "aug", "dim", "dim7"]
        candidates = {}
        for root in roots:
            names = [root + quality for quality in base_qualities if root + quality in USER_CHORD_SHAPES]
            names += [root + quality for quality in base_qualities if root + quality in SECONDARY_CHORD_SHAPES]
            candidates[root] = names
        max_columns = max((len(names) for names in candidates.values()), default=0)
        self.chord_table.setColumnCount(max_columns)
        self.chord_table.horizontalHeader().hide()
        self.chord_table.verticalHeader().hide()
        self.chord_table.setRowCount(len(roots))
        self.chord_table.verticalHeader().setDefaultSectionSize(164)
        for row, root in enumerate(roots):
            for column, name in enumerate(candidates[root]):
                secondary_index = column - len(base_qualities)
                shape = (SECONDARY_CHORD_SHAPES.get(name, [])[secondary_index]
                         if secondary_index >= 0 and secondary_index < len(SECONDARY_CHORD_SHAPES.get(name, []))
                         else chord_shape(name))
                diagram = ChordDiagramButton(name, shape)
                diagram.clicked.connect(lambda checked=False, chord=name, form=shape: self.edit_chord_shape(chord, form))
                self.chord_table.setCellWidget(row, column, diagram)
        self.chord_table.resizeColumnsToContents()
        self.chord_table.blockSignals(False)

    def edit_chord_shape(self, name, shape=None):
        dialog = FretboardPickerDialog(self)
        dialog.set_shape(shape or chord_shape(name))
        if dialog.exec() != QDialog.Accepted:
            return
        selected = getattr(dialog, "selected", None) or name
        shape = USER_CHORD_SHAPES.get(selected)
        if shape:
            self.populate_chord_table()
            self.status.setText(f"コード表を保存しました: {selected} = {shape_to_text(shape)}")
            self.update_chord_label()

    def toggle_chord_table(self):
        visible = not self.chord_table_panel.isVisible()
        self.chord_table_panel.setVisible(visible)
        self.chord_table_button.setText("コード表を隠す" if visible else "コード表")
        if visible and not self.chord_table_loaded:
            self.populate_chord_table()
            self.chord_table_loaded = True

    def search_chord_by_fingering(self):
        dialog = FretboardPickerDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        name = getattr(dialog, "selected", None)
        if name:
            self.status.setText(f"押さえ方の候補: {name}（コード表に登録しました）")
            self.populate_chord_table()

    def on_chord_table_changed(self, row, column):
        if column not in (1, 2):
            return
        name_item = self.chord_table.item(row, 0)
        frets_item = self.chord_table.item(row, 1)
        barre_item = self.chord_table.item(row, 2)
        if not name_item or not frets_item:
            return
        name = name_item.text().strip()
        try:
            if not frets_item.text().strip():
                return
            frets = parse_shape_text(frets_item.text())
            barre_text = (barre_item.text().strip() if barre_item else "")
            barre = None
            if barre_text:
                fret_part, strings_part = barre_text.split(":", 1)
                a, b = strings_part.split("-", 1)
                barre = (int(fret_part), int(a), int(b))
            USER_CHORD_SHAPES[name] = {"frets": frets, "barre": barre}
            save_user_chord_shapes(USER_CHORD_SHAPES)
            self.status.setText(f"コード表を保存しました: {name} = {shape_to_text(USER_CHORD_SHAPES[name])}")
            self.update_chord_label()
            if self.chord_view:
                self.chord_view.update()
            for view in self.acoustic_chord_views:
                view.update()
        except Exception as error:
            self.status.setText(f"コード表の保存に失敗: {error}")

    def update_destination(self):
        self.destination.setText("保存先  " + str(self.output))

    def choose_output(self):
        path = QFileDialog.getExistingDirectory(self, "保存フォルダを選択", str(self.output))
        if path:
            self.output = Path(path)
            self.update_destination()

    def choose_existing_result(self):
        path = QFileDialog.getExistingDirectory(self, "処理済みフォルダを選択", str(self.output))
        if not path:
            return
        folder = Path(path)
        stems = self.detect_stems(folder)
        if not stems:
            self.status.setText("このフォルダには再生できる分離音 .wav が見つかりませんでした。")
            return
        self.render_results(folder, stems)
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.progress_text.setText("100%")
        self.status.setText(folder.name + "：処理済みファイルを読み込みました。")

    def export_audio_video(self):
        if not self.result_folder:
            return
        view_mode, accepted = QInputDialog.getItem(self, "ビューを選択", "書き出すビュー:", ["コードビュー", "ピアノビュー"], 0, False)
        if not accepted:
            return
        mode = "piano" if view_mode == "ピアノビュー" else "chords"
        # A view always travels with its matching separated audio source.
        audio = self.result_folder / ("piano.wav" if mode == "piano" else "acoustic-guitar.wav")
        if not audio or not Path(audio).is_file():
            required = "piano.wav" if mode == "piano" else "acoustic-guitar.wav"
            QMessageBox.warning(self, "音声がありません", f"{required} が見つかりません。先に楽器分離を完了してください。")
            return
        output, _ = QFileDialog.getSaveFileName(self, "書き出し先", str(self.result_folder / "midi_views.mp4"), "MP4動画 (*.mp4)")
        if not output:
            return
        self.export_video.setEnabled(False)
        self.status.setText("MIDIビュー動画を書き出し中…")
        process = QProcess(self)
        self._video_export_process = process
        process.readyReadStandardOutput.connect(lambda p=process: self.on_video_export_output(p))
        process.finished.connect(lambda code, status, p=process: self.on_video_export_finished(p, code, output))
        process.errorOccurred.connect(lambda error, p=process: self.on_video_export_error(p, error))
        process.start(sys.executable, [str(ROOT / "export_midi_video.py"), str(self.result_folder), str(audio), output, mode])

    def on_video_export_output(self, process):
        for line in bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace").splitlines():
            if line.startswith("EXPORT_PROGRESS "):
                value = int(line.split()[-1])
                self.progress.setRange(0, 100)
                self.progress.setValue(value)
                self.progress_text.setText(f"{value}%")

    def on_video_export_finished(self, process, code, output):
        if process is not self._video_export_process:
            return
        self.export_video.setEnabled(True)
        self._video_export_process = None
        if code == 0:
            self.progress.setValue(100)
            self.progress_text.setText("100%")
        self.status.setText(f"音声付き動画を書き出しました: {output}" if code == 0 else "動画の書き出しに失敗しました。")
        process.deleteLater()

    def on_video_export_error(self, process, error):
        if process is self._video_export_process:
            self.export_video.setEnabled(True)
            self._video_export_process = None
            self.status.setText("FFmpegを起動できませんでした。")

    def open_default_result(self):
        folder = self.output / "04 くるみ"
        if self.result_folder is not None or not folder.is_dir():
            return
        stems = self.detect_stems(folder)
        if not stems:
            return
        self.render_results(folder, stems)
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.progress_text.setText("100%")
        self.status.setText(folder.name + "：デバッグ用の処理済みフォルダを読み込みました。")

    def select_source(self, path):
        self.select_sources([path])

    def select_sources(self, paths):
        if self.worker and self.worker.isRunning():
            return
        sources = []
        for path in paths:
            source = Path(path)
            if source.is_file() and source.suffix.lower() in EXTENSIONS:
                sources.append(source.resolve())
        if not sources:
            self.status.setText("対応する音声ファイルを選択してください。")
            return
        self.sources = sources
        if len(sources) == 1:
            source = sources[0]
            self.drop.setText("♫\n" + source.name + "\nクリックまたはドロップで変更")
            self.file_info.setText(f"{source.suffix[1:].upper()}  ·  {source.stat().st_size / 1024**2:.1f} MB")
            self.status.setText("準備完了。モデルを選んで分離を開始してください。")
        else:
            total_size = sum(source.stat().st_size for source in sources)
            self.drop.setText("♫\n" + f"{len(sources)}ファイルを選択中" + "\nクリックまたはドロップで変更")
            self.file_info.setText(f"{len(sources)} files  ·  {total_size / 1024**2:.1f} MB")
            self.status.setText(f"準備完了。{len(sources)}曲を順番に分離します。")
        self.specialist_start.setEnabled(True)

    def update_bpm_label(self):
        for view in (self.chord_view, *self.acoustic_chord_views):
            if view is not None:
                view.set_beat_grid(self.beat_grid)
        if self.acoustic_chord_views and self.acoustic_chords:
            self.show_acoustic_chords(self.acoustic_chords)
        bpm = self.beat_grid.get("bpm") if self.beat_grid else None
        beats = len(self.beat_grid.get("beats", [])) if self.beat_grid else 0
        downbeats = len(self.beat_grid.get("downbeats", [])) if self.beat_grid else 0
        if bpm:
            self.bpm_label.setText(f"BPM: {bpm}  ·  beats {beats}  ·  downbeats {downbeats}")
        else:
            self.bpm_label.setText("BPM: 未推定")

    def update_chord_label(self):
        chords = self.chords.get("chords", []) if self.chords else []
        if not chords:
            if self.chord_view:
                self.chord_view.set_chords({})
            return
        if self.chord_view:
            self.chord_view.set_chords(self.chords)

    def busy(self, active):
        self.drop.setEnabled(not active)
        self.browse_output.setEnabled(not active)
        self.load_existing.setEnabled(not active)
        self.specialist_start.setEnabled(not active and bool(self.sources))
        self.specialist_model.setEnabled(not active)
        self.analyze_prereq.setEnabled(not active and self.result_folder is not None)
        self.analyze_chords.setEnabled(not active and self.result_folder is not None)
        self.cancel.setVisible(active)
        self.cancel.setEnabled(active)

    def clear_results(self):
        self.stop_acoustic_analysis()
        self.player.stop()
        self.player.setSource(QUrl())
        self.stop_vocal_worker()
        self.stop_drum_worker()
        self.stop_piano_worker()
        self.stop_bass_worker()
        self.stop_beat_worker()
        self.stop_full_analysis_worker()
        self.play_buttons.clear()
        self.seek_controls.clear()
        self.pending_seek = None
        self.smooth_position = 0
        self.smooth_anchor = 0
        while self.results.count():
            item = self.results.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.open_output.hide()
        self.drum_view = None
        self.piano_view = None
        self.bass_view = None
        self.chord_view = None
        self.acoustic_chord_view = None
        self.acoustic_chord_views = []
        self.acoustic_chords = {}
        self.drum_note = None
        self.piano_note = None
        self.bass_note = None
        self.beat_grid = {}
        self.chords = {}
        self.update_bpm_label()
        self.update_chord_label()

    def detect_stems(self, folder):
        known_order = []
        for stems in STEMS.values():
            for stem in stems:
                if stem not in known_order:
                    known_order.append(stem)
        existing = {path.stem for path in Path(folder).glob("*.wav") if path.is_file()}
        ordered = [stem for stem in known_order if stem in existing]
        ordered.extend(sorted(existing - set(ordered)))
        return ordered

    def separate(self, checked=False):
        if not self.sources:
            return
        if not shutil.which("ffmpeg"):
            QMessageBox.warning(self, "FFmpegが必要です", "ターミナルで brew install ffmpeg を実行してください。")
            return
        self.clear_results()
        self.logs.clear()
        self.busy(True)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress_text.setText("0%")
        if len(self.sources) == 1:
            self.status.setText("モデルを準備中… 初回はモデルをダウンロードします。")
        else:
            self.status.setText(f"{len(self.sources)}曲を順番に分離します… 初回はモデルをダウンロードします。")
        choice = self.specialist_model.currentData()
        self.worker = Worker(self.sources, self.output, choice)
        self.worker.log.connect(self.on_log)
        self.worker.progress.connect(self.on_progress)
        self.worker.result.connect(self.on_result)
        self.worker.failed.connect(self.on_error)
        self.worker.finished.connect(lambda: self.busy(False))
        self.worker.start()

    def on_log(self, line):
        self.logs.appendPlainText(line)
        if line.startswith("Batch "):
            self.status.setText(line.replace("Batch", "処理"))
        if "Separating track" in line:
            self.status.setText("分離中… 音源の長さによって数分以上かかります。")
        elif "Downloading" in line:
            self.status.setText("初回モデルをダウンロード中… 詳細は処理ログで確認できます。")

    def on_progress(self, value):
        self.progress.setRange(0, 100)
        self.progress.setValue(value)
        self.progress_text.setText(f"{value}%")
        self.status.setText(f"分離中… {value}%")

    def cancel_job(self):
        if self.worker and self.worker.isRunning():
            self.worker.runner.cancel()
            self.cancel.setEnabled(False)
            self.status.setText("キャンセルしています…")

    def on_result(self, folder):
        self.progress.setRange(0, 100)
        if folder is None:
            self.progress.setValue(0)
            self.progress_text.setText("0%")
            self.status.setText("分離をキャンセルしました。")
            return
        self.result_folder = folder
        stems = STEMS[self.worker.count] if self.worker and self.worker.count in STEMS else self.detect_stems(folder)
        self.render_results(folder, stems)
        if self.worker and self.worker.count in SPECIALISTS:
            source_count = len(getattr(self.worker, "sources", []))
            if source_count > 1:
                self.status.setText(f"{SPECIALISTS[self.worker.count][0]}：{source_count}曲の分離が完了しました。最後の曲を表示しています。")
            else:
                self.status.setText(SPECIALISTS[self.worker.count][0] + "：分離が完了しました。")
        else:
            self.status.setText("分離が完了しました。各パートを試聴できます。")

    def render_results(self, folder, stems):
        self.clear_results()
        self.result_folder = Path(folder)
        self.export_video.setEnabled(True)
        beat_cache = self.result_folder / "beat-grid.json"
        if beat_cache.is_file():
            self.beat_grid = load_or_estimate_beats(self.result_folder, force=False)
        chord_cache = self.result_folder / "chords.json"
        if chord_cache.is_file():
            self.chords = load_or_estimate_chords(self.result_folder, force=False)
        self.update_bpm_label()
        self.analyze_prereq.setEnabled(True)
        self.analyze_chords.setEnabled(True)
        self.open_output.show()
        for stem in stems:
            row = QFrame()
            row.setObjectName("stem")
            row.setMinimumHeight(96)
            row_layout = QVBoxLayout(row)
            box = QHBoxLayout()
            row_layout.addLayout(box)
            label = QLabel("●  " + NAMES.get(stem, stem))
            label.setStyleSheet(f"color: {COLORS.get(stem, '#c8c9d4')}; font-weight: 600;")
            box.addWidget(label, 1)
            filename = QLabel(stem + ".wav")
            filename.setObjectName("muted")
            box.addWidget(filename)
            path = self.result_folder / (stem + ".wav")
            if stem in ("vocals", "drums", "piano", "bass") and path.is_file():
                toggle = QPushButton("MIDIビューを表示")
                box.addWidget(toggle)
            elif stem in ("guitar", "acoustic-guitar") and path.is_file():
                toggle = QPushButton("コードビューを表示")
                box.addWidget(toggle)
            else:
                toggle = None
            play = QPushButton("▶ 試聴")
            play.clicked.connect(lambda checked=False, p=path: self.play(p))
            self.play_buttons[str(path)] = play
            box.addWidget(play)
            timeline = QHBoxLayout()
            slider = SeekSlider(Qt.Horizontal)
            try:
                duration = round(sf.info(str(path)).duration * 1000)
            except (OSError, sf.LibsndfileError):
                duration = 0
            slider.setRange(0, duration)
            slider.setEnabled(duration > 0)
            slider.setSingleStep(1000)
            slider.setPageStep(10000)
            slider.setAccessibleName(NAMES.get(stem, stem) + "の再生位置")
            slider.setToolTip("クリック・ドラッグで再生位置を選択（← →：1秒）")
            clock = QLabel(f"0:00 / {timestamp(duration)}")
            clock.setObjectName("muted")
            clock.setMinimumWidth(100)
            self.seek_controls[str(path)] = (slider, clock)
            slider.valueChanged.connect(lambda value, p=path: self.seek(p, value))
            timeline.addWidget(slider, 1)
            timeline.addWidget(clock)
            row_layout.addLayout(timeline)
            if stem == "vocals" and path.is_file():
                self.add_vocal_panel(row_layout, toggle)
            elif stem == "drums" and path.is_file():
                self.add_drum_panel(row_layout, toggle)
            elif stem == "piano" and path.is_file():
                self.add_piano_panel(row_layout, toggle)
            elif stem == "bass" and path.is_file():
                self.add_bass_panel(row_layout, toggle)
            elif stem == "guitar" and path.is_file():
                self.add_acoustic_chord_panel(row_layout, toggle, default_visible=False)
            elif stem == "acoustic-guitar" and path.is_file():
                self.add_acoustic_chord_panel(row_layout, toggle, default_visible=True)
            self.results.addWidget(row)
        self.update_chord_label()

    def edit_chord(self, index, current, range_start=None, range_end=None, acoustic=False):
        target_data = self.acoustic_chords if acoustic else self.chords
        chords = target_data.get("chords", []) if target_data else []
        if index < 0 or index >= len(chords):
            return
        dialog = ChordCandidateDialog(current, chord_edit_suggestions(current), self)
        if dialog.exec() != QDialog.Accepted or not dialog.selected:
            return
        value = dialog.selected.strip()
        start, end = self.manual_chord_range(range_start, range_end, chords[index])
        replacement = {"start": start, "end": end, "chord": value, "confidence": 1.0, "edited": True}
        if dialog.selected_shape is not None:
            replacement["shape"] = dialog.selected_shape
        updated, inserted = [], False
        # Replace precisely one quarter-note cell, splitting the surrounding
        # detected chord only when the clicked cell falls inside it.
        for item in chords:
            item_start, item_end = float(item.get("start", 0)), float(item.get("end", 0))
            if item_end <= start + 1e-7 or item_start >= end - 1e-7:
                updated.append(dict(item))
                continue
            if item_start < start - 1e-7:
                left = dict(item)
                left["end"] = start
                updated.append(left)
            if not inserted:
                updated.append(replacement)
                inserted = True
            if item_end > end + 1e-7:
                right = dict(item)
                right["start"] = end
                updated.append(right)
        if not inserted:
            updated.append(replacement)
        # Keep neighbouring equal-labelled segments distinct.  They can be
        # separate bars deliberately, and editing the second one must never
        # turn the first into the same manual correction.
        chords[:] = sorted(updated, key=lambda item: (float(item["start"]), float(item["end"])))
        message = f"コードを修正しました: {timestamp(start * 1000)}〜{timestamp(end * 1000)} を {value} に変更"
        target_data["chords"] = sorted(chords, key=lambda item: (float(item.get("start", 0)), float(item.get("end", 0))))
        if acoustic:
            # Manual ranges are authoritative.  Do not run the acoustic
            # short-segment repair here: it is allowed to extend a neighbour
            # across a one-beat cell and would undo the user's insertion.
            self.acoustic_chords = target_data
            target_data["_manual_edits"] = True
            target_data["_raw_chords"] = [dict(item) for item in target_data["chords"]]
            if self.result_folder:
                (self.result_folder / ACOUSTIC_CHORD_CACHE).write_text(
                    json.dumps(target_data, ensure_ascii=False, indent=2), encoding="utf-8")
            if self.acoustic_chord_view:
                for view in self.acoustic_chord_views:
                    view.set_chords(target_data)
        else:
            self.chords = target_data
            self.save_chords()
            self.update_chord_label()
        self.status.setText(message)

    def snap_chord_time(self, value):
        """Snap manually entered chord positions to an eighth-note grid."""
        beats = sorted({float(t) for t in self.beat_grid.get("beats", [])
                        if np.isfinite(float(t))})
        candidates = list(beats)
        if len(beats) >= 2:
            # Keep the detected beat locations and add each inter-beat midpoint
            # so tempo drift in the source does not accumulate across the song.
            candidates.extend((left + right) / 2 for left, right in zip(beats, beats[1:]))
        else:
            bpm = self.beat_grid.get("bpm")
            if bpm and float(bpm) > 0:
                half_beat = 30.0 / float(bpm)
                anchor = beats[0] if beats else 0.0
                center = round((float(value) - anchor) / half_beat)
                return max(0.0, anchor + center * half_beat)
        if not candidates:
            return max(0.0, float(value))
        return max(0.0, min(candidates, key=lambda t: abs(t - float(value))))

    def chord_step_seconds(self, start):
        """Return one quarter-note duration near a chord's start."""
        beats = sorted(float(t) for t in self.beat_grid.get("beats", [])
                       if np.isfinite(float(t)))
        intervals = np.diff(beats) if len(beats) >= 2 else np.array([])
        valid = intervals[(intervals > 0.2) & (intervals < 2.5)]
        if valid.size:
            return float(np.median(valid))
        bpm = self.beat_grid.get("bpm")
        return 60.0 / float(bpm) if bpm and float(bpm) > 0 else 0.5

    def move_chord_timing(self, chords, index, delta):
        """Move a chord by a musical subdivision while preserving adjacency."""
        chords.sort(key=lambda item: float(item.get("start", 0)))
        item = chords[index]
        start = float(item.get("start", 0))
        end = float(item.get("end", start + 1))
        duration = max(0.05, end - start)
        new_start = max(0.0, start + float(delta))
        previous_end = float(chords[index - 1].get("end", 0)) if index > 0 else 0.0
        next_start = float(chords[index + 1].get("start", new_start + duration)) if index + 1 < len(chords) else None
        new_start = max(previous_end, new_start)
        if next_start is not None:
            new_start = min(new_start, max(previous_end, next_start - duration))
        if abs(new_start - start) < 1e-6:
            return None
        item["start"] = round(new_start, 3)
        item["end"] = round(new_start + duration, 3)
        if index > 0:
            chords[index - 1]["end"] = round(new_start, 3)
        if index + 1 < len(chords):
            chords[index + 1]["start"] = round(new_start + duration, 3)
        item["edited"] = True
        return new_start

    def manual_chord_range(self, range_start, range_end, target):
        """Use complete beat-grid cells supplied by a click or drag selection."""
        if range_start is not None and range_end is not None:
            return round(float(range_start), 6), round(float(range_end), 6)
        beats = sorted({float(t) for t in self.beat_grid.get("beats", []) if np.isfinite(float(t))})
        if len(beats) >= 2:
            start = float(target.get("start", 0))
            for left, right in zip(beats, beats[1:]):
                if left <= start < right:
                    return round(left, 6), round(right, 6)
        return float(target.get("start", 0)), float(target.get("end", 0))

    @staticmethod
    def merge_manual_chords(chords):
        merged = []
        for chord in sorted(chords, key=lambda item: (float(item["start"]), float(item["end"]))):
            if (merged and merged[-1].get("chord") == chord.get("chord") and
                    abs(float(merged[-1]["end"]) - float(chord["start"])) < 1e-6):
                merged[-1]["end"] = chord["end"]
                merged[-1]["edited"] = True
            else:
                merged.append(chord)
        return merged

    def edit_acoustic_chord(self, index, current, range_start, range_end):
        self.edit_chord(index, current, range_start, range_end, acoustic=True)

    def save_chords(self):
        if self.result_folder:
            (self.result_folder / "chords.json").write_text(json.dumps(self.chords, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_vocal_panel(self, row_layout, toggle):
        panel = QWidget()
        panel.hide()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 8, 0, 0)
        self.vocal_view = VocalRollView()
        if self.beat_grid:
            self.vocal_view.set_beat_grid(self.beat_grid)
        layout.addWidget(self.vocal_view)
        cache = self.result_folder / "vocals-midi.json"
        if cache.is_file():
            note_text = "保存済みのボーカルMIDI推定を読み込み中…"
        else:
            note_text = "Basic Pitchで vocals.wav を解析できます。"
        note = QLabel(note_text)
        self.vocal_note = note
        note.setObjectName("muted")
        note.setWordWrap(True)
        layout.addWidget(note)
        row_layout.addWidget(panel)
        if toggle:
            toggle.clicked.connect(lambda checked=False, p=panel, b=toggle: self.toggle_midi_panel(p, b))
        if cache.is_file():
            self.on_vocal_midi(load_or_estimate_vocal(self.result_folder, force=False), note, None)

    def add_acoustic_chord_panel(self, row_layout, toggle, default_visible=True):
        panel = QWidget()
        if not default_visible:
            panel.hide()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 8, 0, 0)
        self.acoustic_chord_view = AcousticChordView()
        self.acoustic_chord_views.append(self.acoustic_chord_view)
        self.acoustic_chord_view.set_beat_grid(self.beat_grid)
        self.acoustic_chord_view.chordEdited.connect(self.edit_acoustic_chord)
        layout.addWidget(self.acoustic_chord_view)
        controls = QHBoxLayout()
        self.acoustic_chord_note = QLabel("Solititoでアコギのコードを推定できます。")
        self.acoustic_chord_note.setObjectName("muted")
        self.acoustic_chord_note.setWordWrap(True)
        controls.addWidget(self.acoustic_chord_note, 1)
        layout.addLayout(controls)
        row_layout.addWidget(panel)
        if toggle:
            toggle.setText("コードビューを隠す" if default_visible else "コードビューを表示")
            toggle.clicked.connect(lambda checked=False, p=panel, b=toggle: self.toggle_chord_panel(p, b))
        cache = self.result_folder / ACOUSTIC_CHORD_CACHE
        if cache.is_file():
            try:
                self.show_acoustic_chords(json.loads(cache.read_text(encoding="utf-8")))
            except (ValueError, OSError, KeyError, TypeError) as error:
                self.acoustic_chord_note.setText(f"保存済みコードを読み込めません: {error}")

    def show_acoustic_chords(self, data):
        # The normal inference result needs repair/quantisation. Once a user
        # has edited a beat cell, its exact boundaries must survive refreshes
        # and application restarts, so do not run that repair again.
        if not data.get("_manual_edits"):
            data = prepare_chords(data, self.beat_grid)
        self.acoustic_chords = data
        for view in self.acoustic_chord_views:
            view.set_chords(data)
            view.set_position_ms(self.smooth_position)
        self.acoustic_chord_note.setText(f"Solitito · {len(data.get('chords', []))} コード区間（参考） · 短い区間を補完・4分音符に整列")

    def load_acoustic_chords(self, checked=False):
        if (not self.result_folder or self.acoustic_process is not None or
                (self.full_analysis_worker and self.full_analysis_worker.isRunning())):
            return
        self.acoustic_chord_note.setText("Solititoで再コード解析中…（毎回再実行）")
        process = QProcess(self)
        self.acoustic_process = process
        process.setWorkingDirectory(str(ROOT))
        process.finished.connect(lambda code, status, p=process: self.on_acoustic_finished(p, code))
        process.errorOccurred.connect(lambda error, p=process: self.on_acoustic_process_error(p, error))
        process.start(sys.executable, [str(ROOT / "analysis" / "acoustic_chords.py"), str(self.result_folder.resolve()), "--force"])

    def on_acoustic_process_error(self, process, error):
        if process is self.acoustic_process:
            self.acoustic_chord_note.setText("アコギのコード推定に失敗しました: " + process.errorString())
            self.acoustic_process = None
            process.deleteLater()

    def on_acoustic_finished(self, process, code):
        if process is not self.acoustic_process:
            return
        succeeded = False
        try:
            if code != 0:
                raise RuntimeError(bytes(process.readAllStandardError()).decode("utf-8", errors="replace")[-1500:])
            self.show_acoustic_chords(json.loads(bytes(process.readAllStandardOutput()).decode("utf-8")))
            succeeded = True
        except (ValueError, RuntimeError, KeyError, TypeError) as error:
            self.acoustic_chord_note.setText(f"アコギのコード推定に失敗しました: {error}")
        finally:
            self.acoustic_process = None
            process.deleteLater()
        if (succeeded and self.result_folder and
                any((self.result_folder / name).is_file()
                    for name in ("piano-midi.json", "bass-midi.json", "drums-midi.json"))):
            # Keep the electric+acoustic timeline on the same fresh acoustic
            # evidence instead of leaving its previous integrated result visible.
            self.load_chord_analysis()

    def stop_acoustic_analysis(self):
        process = self.acoustic_process
        self.acoustic_process = None
        if process is not None:
            process.kill()
            process.waitForFinished(1000)
            process.deleteLater()

    def add_drum_panel(self, row_layout, toggle):
        panel = QWidget()
        panel.hide()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 8, 0, 0)
        self.drum_view = DrumMidiView()
        if self.beat_grid:
            self.drum_view.set_beat_grid(self.beat_grid)
        layout.addWidget(self.drum_view)
        cache = self.result_folder / "drums-midi.json"
        if cache.is_file():
            note_text = "保存済みのドラムMIDI推定を読み込み中…"
        else:
            note_text = "ADTOF-pytorchで drums.wav を解析できます。解析後、再生位置に同期して表示します。"
        if self.beat_grid:
            bpm = self.beat_grid.get("bpm")
            bpm_text = f"{bpm} BPM" if bpm else "BPM不明"
            note_text += f"\n拍グリッド読込済み: {bpm_text}"
        note = QLabel(note_text)
        self.drum_note = note
        note.setObjectName("muted")
        note.setWordWrap(True)
        layout.addWidget(note)
        row_layout.addWidget(panel)
        analyze = None
        if toggle:
            toggle.clicked.connect(lambda checked=False, p=panel, b=toggle: self.toggle_midi_panel(p, b))
        if cache.is_file():
            self.on_drum_midi(load_or_estimate(self.result_folder, force=False), note, analyze)


    def add_bass_panel(self, row_layout, toggle):
        panel = QWidget()
        panel.hide()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 8, 0, 0)

        self.bass_view = BassRollView()
        if self.beat_grid:
            self.bass_view.set_beat_grid(self.beat_grid)
        layout.addWidget(self.bass_view)

        cache = self.result_folder / "bass-midi.json"
        if cache.is_file():
            note_text = "保存済みのベースMIDI推定を読み込み中…"
        else:
            note_text = "Basic Pitchで bass.wav を解析できます。"
        if self.beat_grid:
            bpm = self.beat_grid.get("bpm")
            bpm_text = f"{bpm} BPM" if bpm else "BPM不明"
            note_text += f"\n拍グリッド読込済み: {bpm_text}"
        note = QLabel(note_text)
        self.bass_note = note
        note.setObjectName("muted")
        note.setWordWrap(True)
        layout.addWidget(note)

        row_layout.addWidget(panel)
        analyze = None

        if toggle:
            toggle.clicked.connect(lambda checked=False, p=panel, b=toggle: self.toggle_midi_panel(p, b))

        if cache.is_file():
            self.on_bass_midi(load_or_estimate_bass(self.result_folder, force=False), note, analyze)

    def add_piano_panel(self, row_layout, toggle):
        panel = QWidget()
        panel.hide()

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 8, 0, 0)

        self.piano_view = PianoRollView()
        if self.beat_grid:
            self.piano_view.set_beat_grid(self.beat_grid)
        layout.addWidget(self.piano_view)

        cache = self.result_folder / "piano-midi.json"

        if cache.is_file():
            note_text = "保存済みのピアノMIDI推定を読み込み中…"
        else:
            note_text = "Transkun V2で piano.wav を解析できます。"
        if self.beat_grid:
            bpm = self.beat_grid.get("bpm")
            bpm_text = f"{bpm} BPM" if bpm else "BPM不明"
            note_text += f"\n拍グリッド読込済み: {bpm_text}"

        note = QLabel(note_text)
        self.piano_note = note
        note.setObjectName("muted")
        note.setWordWrap(True)
        layout.addWidget(note)

        row_layout.addWidget(panel)
        analyze = None

        if toggle:
            toggle.clicked.connect(
                lambda checked=False, p=panel, b=toggle:
                self.toggle_midi_panel(p, b)
            )

        if cache.is_file():
            self.on_piano_midi(load_or_estimate_piano(self.result_folder, force=False), note, analyze)

    def toggle_midi_panel(self, panel, button):
        visible = not panel.isVisible()
        panel.setVisible(visible)
        button.setText("MIDIビューを隠す" if visible else "MIDIビューを表示")

    def toggle_chord_panel(self, panel, button):
        visible = not panel.isVisible()
        panel.setVisible(visible)
        button.setText("コードビューを隠す" if visible else "コードビューを表示")

    def change_playback_rate(self):
        rate = self.speed_combo.currentData()
        self.player.setPlaybackRate(float(rate or 1.0))
        self.smooth_anchor = self.player.position()
        self.smooth_clock.restart()

    def play(self, path):
        url = QUrl.fromLocalFile(str(path))
        if self.player.source() == url and self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            if self.player.source() != url:
                slider, _ = self.seek_controls[str(path)]
                self.pending_seek = (str(path), slider.value())
                self.player.setSource(url)
            elif self.player.mediaStatus() == QMediaPlayer.EndOfMedia:
                self.player.setPosition(0)
            self.player.play()
        self.sync_play_buttons(self.player.playbackState())

    def seek(self, path, position):
        slider, clock = self.seek_controls[str(path)]
        clock.setText(f"{timestamp(position)} / {timestamp(slider.maximum())}")
        if self.player.source().toLocalFile() == str(path):
            if self.pending_seek:
                self.pending_seek = (str(path), position)
            else:
                self.player.setPosition(position)

    def media_ready(self, status):
        if status in (QMediaPlayer.LoadedMedia, QMediaPlayer.BufferedMedia) and self.pending_seek:
            path, position = self.pending_seek
            if self.player.source().toLocalFile() == path:
                self.pending_seek = None
                self.player.setPosition(position)

    def update_play_position(self, position):
        self.smooth_position = max(0, int(position))
        self.smooth_anchor = self.smooth_position
        self.smooth_clock.restart()
        self.render_playback_position(self.smooth_position)

    def render_playback_position(self, position):
        if self.vocal_view:
            self.vocal_view.set_position_ms(position)
        if self.drum_view:
            self.drum_view.set_position_ms(position + self.drum_visual_offset_ms)
        if self.piano_view:
            self.piano_view.set_position_ms(position)
        if self.bass_view:
            self.bass_view.set_position_ms(position)
        if self.chord_view:
            self.chord_view.set_position_ms(position)
        if self.acoustic_chord_view:
            for view in self.acoustic_chord_views:
                view.set_position_ms(position)
        controls = self.seek_controls.get(self.player.source().toLocalFile())
        if not controls or self.pending_seek:
            return
        slider, clock = controls
        if not slider.isSliderDown():
            slider.blockSignals(True)
            slider.setValue(position)
            slider.blockSignals(False)
            clock.setText(f"{timestamp(position)} / {timestamp(slider.maximum())}")

    def refresh_playback_position(self):
        if self.player.playbackState() == QMediaPlayer.PlayingState and self.smooth_clock.isValid():
            position = self.smooth_anchor + self.smooth_clock.elapsed() * self.player.playbackRate()
            duration = self.player.duration()
            if duration > 0:
                position = min(position, duration)
            self.render_playback_position(position)
        else:
            self.update_play_position(self.player.position())

    def sync_play_buttons(self, state):
        if state == QMediaPlayer.PlayingState:
            self.smooth_anchor = self.player.position()
            self.smooth_clock.restart()
            if not self.playback_timer.isActive():
                self.playback_timer.start()
        else:
            self.playback_timer.stop()
            self.refresh_playback_position()
        for path, button in self.play_buttons.items():
            active = state == QMediaPlayer.PlayingState and self.player.source().toLocalFile() == path
            button.setText("Ⅱ 一時停止" if active else "▶ 試聴")

    def on_error(self, message):
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress_text.setText("0%")
        self.status.setText("分離できませんでした。ログを確認し、再実行してください。")
        self.logs.appendPlainText(message)
        self.logs.show()

    def load_vocal_midi(self, folder, note, button=None, force=False):
        if self.vocal_worker and self.vocal_worker.isRunning():
            self.stop_vocal_worker()
        if button:
            button.setEnabled(False)
        note.setText("Basic Pitchで解析中…")
        self.vocal_worker = VocalMidiWorker(folder, force=force)
        self.vocal_worker.result.connect(lambda events, n=note, b=button: self.on_vocal_midi(events, n, b))
        self.vocal_worker.failed.connect(lambda message, n=note, b=button: self.on_vocal_midi_error(message, n, b))
        self.vocal_worker.start()

    def stop_vocal_worker(self):
        if self.vocal_worker and self.vocal_worker.isRunning():
            self.vocal_worker.requestInterruption()
            if not self.vocal_worker.wait(1000):
                self.vocal_worker.terminate()
                self.vocal_worker.wait(1000)

    def on_vocal_midi(self, events, note, button=None):
        if self.vocal_view:
            self.vocal_view.set_events(events)
        count = len(events.get("notes", []))
        raw = events.get("_raw_notes")
        suffix = f" / raw {raw}" if raw is not None else ""
        note.setText(f"{events.get('_model', 'Basic Pitch vocal melody')} で推定: Notes {count}{suffix}")
        if button:
            button.setEnabled(True)

    def on_vocal_midi_error(self, message, note, button=None):
        note.setText("ボーカルMIDI推定に失敗しました: " + message)
        if button:
            button.setEnabled(True)

    def load_full_analysis(self, force=False):
        if not self.result_folder:
            return
        if self.full_analysis_worker and self.full_analysis_worker.isRunning():
            return
        existing_analysis = any((self.result_folder / filename).is_file() for filename in (
            "beat-grid.json", "vocals-midi.json", "drums-midi.json", "piano-midi.json", "bass-midi.json"))
        if existing_analysis:
            answer = QMessageBox.question(
                self, "BPM/MIDI解析の確認",
                "BPM、MIDI解析を再実行しますか？既存の解析ファイルは上書きされます。",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return
            force = True
        self.analyze_prereq.setEnabled(False)
        self.analyze_chords.setEnabled(False)
        self.status.setText("BPM/MIDI推定を開始します…（保存済みファイルは再利用）")
        self.full_analysis_worker = FullAnalysisWorker(self.result_folder, force=force)
        self.full_analysis_worker.status.connect(self.status.setText)
        self.full_analysis_worker.result.connect(self.on_full_analysis)
        self.full_analysis_worker.failed.connect(self.on_full_analysis_error)
        self.full_analysis_worker.finished.connect(lambda: self._analysis_buttons_ready())
        self.full_analysis_worker.start()

    def load_chord_analysis(self, checked=False):
        if not self.result_folder:
            return
        if self.full_analysis_worker and self.full_analysis_worker.isRunning():
            return
        if (self.result_folder / "chords.json").is_file():
            answer = QMessageBox.question(
                self, "コード解析の確認",
                "コード解析を行いますか？手動で行ったコード修正は上書きされます。",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return
        self.analyze_prereq.setEnabled(False)
        self.analyze_chords.setEnabled(False)
        self.status.setText("コード解析を開始します…（毎回再計算）")
        self.full_analysis_worker = ChordAnalysisWorker(self.result_folder)
        self.full_analysis_worker.status.connect(self.status.setText)
        self.full_analysis_worker.result.connect(self.on_full_analysis)
        self.full_analysis_worker.failed.connect(self.on_full_analysis_error)
        self.full_analysis_worker.finished.connect(lambda: self._analysis_buttons_ready())
        self.full_analysis_worker.start()

    def _analysis_buttons_ready(self):
        enabled = self.result_folder is not None
        self.analyze_prereq.setEnabled(enabled)
        self.analyze_chords.setEnabled(enabled)

    def stop_full_analysis_worker(self):
        if self.full_analysis_worker and self.full_analysis_worker.isRunning():
            self.full_analysis_worker.requestInterruption()
            if not self.full_analysis_worker.wait(1000):
                self.full_analysis_worker.terminate()
                self.full_analysis_worker.wait(1000)

    def on_full_analysis(self, result):
        self.beat_grid = result.get("beat_grid", self.beat_grid)
        if self.vocal_view:
            self.vocal_view.set_beat_grid(self.beat_grid)
            if "vocals" in result:
                self.vocal_view.set_events(quantize_note_events(result["vocals"], self.beat_grid))
                if self.vocal_note:
                    count = len(result["vocals"].get("notes", []))
                    raw = result["vocals"].get("_raw_notes")
                    suffix = f" / raw {raw}" if raw is not None else ""
                    self.vocal_note.setText(f"{result['vocals'].get('_model', 'Basic Pitch vocal melody')} で推定: Notes {count}{suffix}")
        if self.drum_view:
            self.drum_view.set_beat_grid(self.beat_grid)
            if "drums" in result:
                self.drum_view.set_events(quantize_drum_events(result["drums"], self.beat_grid))
                if self.drum_note:
                    model = result["drums"].get("_model", "ADTOF-pytorch")
                    counts = " / ".join(f"{DRUM_NAMES[lane]} {len(result['drums'].get(lane, []))}" for lane in DRUM_DISPLAY_LANES)
                    self.drum_note.setText(f"{model} で推定: {counts}")
        if self.piano_view:
            self.piano_view.set_beat_grid(self.beat_grid)
            if "piano" in result:
                self.piano_view.set_events(quantize_note_events(result["piano"], self.beat_grid, preserve_end=True))
                if self.piano_note:
                    self.piano_note.setText(f"{result['piano'].get('_model', 'Transkun V2')} で推定: Notes {len(result['piano'].get('notes', []))}")
        if self.bass_view:
            self.bass_view.set_beat_grid(self.beat_grid)
            if "bass" in result:
                self.bass_view.set_events(quantize_note_events(result["bass"], self.beat_grid))
                if self.bass_note:
                    count = len(result["bass"].get("notes", []))
                    raw = result["bass"].get("_raw_notes")
                    suffix = f" / raw {raw}" if raw is not None else ""
                    self.bass_note.setText(f"{result['bass'].get('_model', 'Basic Pitch')} で推定: Notes {count}{suffix}")
        self.chords = result.get("chords", {})
        if "acoustic" in result and self.acoustic_chord_view:
            self.show_acoustic_chords(result["acoustic"])
        self.update_bpm_label()
        self.update_chord_label()
        chord_count = len(self.chords.get("chords", [])) if self.chords else 0
        if "chords" in result:
            self.status.setText(f"コード解析が完了しました。コード {chord_count} 区間")
        else:
            self.status.setText("BPM/MIDI/アコギ解析が完了しました。保存済みファイルは再利用しました。")

    def on_full_analysis_error(self, message):
        self._analysis_buttons_ready()
        self.status.setText("解析に失敗しました。")
        self.logs.appendPlainText(message)
        self.logs.show()

    def load_drum_midi(self, folder, note, button=None, force=False):
        if self.drum_worker and self.drum_worker.isRunning():
            self.stop_vocal_worker()
            self.stop_drum_worker()
        if button:
            button.setEnabled(False)
        note.setText("ADTOF-pytorchで解析中…")
        self.drum_worker = DrumMidiWorker(folder, force=force)
        self.drum_worker.result.connect(lambda events, n=note, b=button: self.on_drum_midi(events, n, b))
        self.drum_worker.failed.connect(lambda message, n=note, b=button: self.on_drum_midi_error(message, n, b))
        self.drum_worker.start()

    def stop_drum_worker(self):
        if self.drum_worker and self.drum_worker.isRunning():
            self.drum_worker.requestInterruption()
            if not self.drum_worker.wait(1000):
                self.drum_worker.terminate()
                self.drum_worker.wait(1000)

    def load_piano_midi(self, folder, note, button=None, force=False):
        if self.piano_worker and self.piano_worker.isRunning():
            self.stop_piano_worker()
        if button:
            button.setEnabled(False)
        note.setText("Transkun V2で解析中…")
        self.piano_worker = PianoMidiWorker(folder, force=force)
        self.piano_worker.result.connect(lambda events, n=note, b=button: self.on_piano_midi(events, n, b))
        self.piano_worker.failed.connect(lambda message, n=note, b=button: self.on_piano_midi_error(message, n, b))
        self.piano_worker.start()

    def stop_piano_worker(self):
        if self.piano_worker and self.piano_worker.isRunning():
            self.piano_worker.requestInterruption()
            if not self.piano_worker.wait(1000):
                self.piano_worker.terminate()
                self.piano_worker.wait(1000)

    def on_piano_midi(self, events, note, button=None):
        if self.piano_view:
            self.piano_view.set_events(events)
        note.setText(f"{events.get('_model', 'Transkun V2')} で推定: Notes {len(events.get('notes', []))}")
        if button:
            button.setEnabled(True)

    def on_piano_midi_error(self, message, note, button=None):
        note.setText("ピアノMIDI推定に失敗しました: " + message)
        if button:
            button.setEnabled(True)


    def load_bass_midi(self, folder, note, button=None, force=False):
        if self.bass_worker and self.bass_worker.isRunning():
            self.stop_bass_worker()
        if button:
            button.setEnabled(False)
        note.setText("Basic Pitchで解析中…")
        self.bass_worker = BassMidiWorker(folder, force=force)
        self.bass_worker.result.connect(lambda events, n=note, b=button: self.on_bass_midi(events, n, b))
        self.bass_worker.failed.connect(lambda message, n=note, b=button: self.on_bass_midi_error(message, n, b))
        self.bass_worker.start()

    def stop_bass_worker(self):
        if self.bass_worker and self.bass_worker.isRunning():
            self.bass_worker.requestInterruption()
            if not self.bass_worker.wait(1000):
                self.bass_worker.terminate()
                self.bass_worker.wait(1000)

    def on_bass_midi(self, events, note, button=None):
        if self.bass_view:
            self.bass_view.set_events(events)
        count = len(events.get("notes", []))
        raw = events.get("_raw_notes")
        suffix = f" / raw {raw}" if raw is not None else ""
        note.setText(f"{events.get('_model', 'Basic Pitch')} で推定: Notes {count}{suffix}")
        if button:
            button.setEnabled(True)

    def on_bass_midi_error(self, message, note, button=None):
        note.setText("ベースMIDI推定に失敗しました: " + message)
        if button:
            button.setEnabled(True)

    def load_beat_grid(self, folder, note, button=None, force=False):
        if self.beat_worker and self.beat_worker.isRunning():
            self.stop_beat_worker()
        if button:
            button.setEnabled(False)
        note.setText("Beat This!でBPM/拍グリッドを解析中…")
        self.beat_worker = BeatGridWorker(folder, force=force)
        self.beat_worker.result.connect(lambda grid, n=note, b=button: self.on_beat_grid(grid, n, b))
        self.beat_worker.failed.connect(lambda message, n=note, b=button: self.on_beat_grid_error(message, n, b))
        self.beat_worker.start()

    def stop_beat_worker(self):
        if self.beat_worker and self.beat_worker.isRunning():
            self.beat_worker.requestInterruption()
            if not self.beat_worker.wait(1000):
                self.beat_worker.terminate()
                self.beat_worker.wait(1000)

    def on_beat_grid(self, grid, note, button=None):
        self.beat_grid = grid
        if self.vocal_view:
            self.vocal_view.set_beat_grid(grid)
        if self.drum_view:
            self.drum_view.set_beat_grid(grid)
            self.drum_view.set_events(quantize_drum_events(self.drum_view.events, grid))
        if self.piano_view:
            self.piano_view.set_beat_grid(grid)
        if self.bass_view:
            self.bass_view.set_beat_grid(grid)
        bpm = grid.get("bpm")
        count = len(grid.get("beats", []))
        downbeats = len(grid.get("downbeats", []))
        bpm_text = f"{bpm} BPM" if bpm else "BPM不明"
        self.update_bpm_label()
        note.setText(f"Beat This!で推定: {bpm_text} / beats {count} / downbeats {downbeats}")
        if button:
            button.setEnabled(True)

    def on_beat_grid_error(self, message, note, button=None):
        note.setText("BPM/拍グリッド推定に失敗しました: " + message)
        if button:
            button.setEnabled(True)

    def on_drum_midi(self, events, note, button=None):
        if self.drum_view:
            if self.beat_grid:
                self.drum_view.set_events(quantize_drum_events(events, self.beat_grid))
            else:
                self.drum_view.set_events(events)
        model = events.get("_model", "ADTOF-pytorch")
        counts = " / ".join(f"{DRUM_NAMES[lane]} {len(events.get(lane, []))}" for lane in DRUM_DISPLAY_LANES)
        note.setText(f"{model} で推定: {counts}")
        if button:
            button.setEnabled(True)

    def on_drum_midi_error(self, message, note, button=None):
        note.setText("ドラムMIDI推定に失敗しました: " + message)
        if button:
            button.setEnabled(True)

    def toggle_log(self):
        visible = not self.logs.isVisible()
        self.logs.setVisible(visible)
        self.log_toggle.setText("処理ログを隠す" if visible else "処理ログを表示")

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.runner.cancel()
            self.status.setText("処理を停止しています。停止後に閉じてください。")
            event.ignore()
        else:
            self.stop_acoustic_analysis()
            self.stop_drum_worker()
            self.stop_piano_worker()
            self.stop_beat_worker()
            self.stop_full_analysis_worker()
            self.player.stop()
            self.playback_timer.stop()
            event.accept()


STYLE = """
QWidget { background: #14161c; color: #ebeaf1; font-family: 'Hiragino Sans'; font-size: 12px; }
QScrollArea { border: none; }
QComboBox { padding: 9px 12px; background: #242731; border: 1px solid #363946; border-radius: 8px; }
QLabel#title { font-size: 30px; font-weight: 700; }
QLabel#eyebrow { color: #ac9be9; font-size: 10px; font-weight: 600; }
QLabel#muted { color: #999baa; font-size: 11px; }
QLabel#progressText { color: #d8cffb; font-size: 12px; font-weight: 700; }
QPushButton { background: #242731; border: 1px solid #363946; border-radius: 8px; padding: 9px 15px; }
QPushButton:hover { background: #303440; border-color: #82739f; }
QPushButton:disabled { color: #656776; border-color: #2a2c36; background: #20222a; }
QPushButton#mode { text-align: left; padding: 15px; font-size: 12px; }
QPushButton#mode:checked { border: 1px solid #b69bff; background: #30283f; color: #e5d8ff; }
QPushButton#drop { border: 1px dashed #666078; background: #1c1e27; font-size: 15px; padding: 24px; }
QPushButton#drop:hover, QPushButton#drop[dragging=true] { background: #2b2638; border-color: #c0abff; }
QPushButton#primary { background: #c1a9ff; color: #20172f; font-size: 14px; font-weight: 700; border: none; }
QPushButton#primary:hover { background: #d0bfff; }
QPushButton#primary:disabled { background: #393247; color: #8e819e; }
QProgressBar { border: none; background: #2b2d37; border-radius: 5px; }
QProgressBar::chunk { background: #bea6ff; border-radius: 5px; }
QFrame#stem { background: #1e212a; border: 1px solid #30333d; border-radius: 8px; }
QFrame#stem QLabel { background: transparent; }
QSlider { background: transparent; min-height: 22px; }
QSlider::groove:horizontal { height: 4px; background: #414451; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #bea6ff; border-radius: 2px; }
QSlider::handle:horizontal { background: #d8c8ff; width: 12px; margin: -4px 0; border-radius: 6px; }
QSlider::handle:horizontal:hover { background: #ffffff; }
QPlainTextEdit { background: #101116; border: 1px solid #30333d; color: #aaadbb; font-family: monospace; font-size: 10px; }
"""


if __name__ == "__main__":
    os.environ["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + os.environ.get("PATH", "")
    app = QApplication(sys.argv)
    app.setApplicationName("STEM Studio")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = Window()
    window.show()
    sys.exit(app.exec())
