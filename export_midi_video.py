"""Render analysis JSON directly to a fixed-layout MIDI video."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT, FPS, WINDOW = 1280, 720, 30, 8.0
WHITE_PCS = (0, 2, 4, 5, 7, 9, 11)
WHITE_INDEX = {pitch_class: index for index, pitch_class in enumerate(WHITE_PCS)}
# Keep 41 white keys, but move the displayed keyboard one octave upward.
# This removes the leftmost octave and adds one at the right; MIDI bars thus
# land one octave farther left on screen.  G1–E7.
KEYBOARD_LOW, KEYBOARD_HIGH, KEYBOARD_BASE_WHITE = 19, 88, 11


def piano_key_geometry(pitch: int, left: float, white_width: float) -> tuple[float, float, bool]:
    """Return x, key width and black-key flag using a conventional piano layout."""
    octave, pitch_class = divmod(pitch, 12)
    white = octave * 7 + WHITE_INDEX.get(pitch_class, 0) - KEYBOARD_BASE_WHITE
    x = left + white * white_width
    if pitch_class in (1, 3, 6, 8, 10):
        # Black keys sit between their neighbouring white keys, not on an even grid.
        preceding_white = {1: 0, 3: 1, 6: 3, 8: 4, 10: 5}[pitch_class]
        x = left + (octave * 7 + preceding_white + .72 - KEYBOARD_BASE_WHITE) * white_width
        return x, white_width * .56, True
    return x, white_width, False


def draw_falling_piano(draw: ImageDraw.ImageDraw, notes: list[dict], time: float, font) -> None:
    """Draw falling MIDI notes and a proportioned 61-key piano keyboard."""
    left, right, top, key_top, bottom = 50, WIDTH - 50, 58, 552, 682
    low, high = KEYBOARD_LOW, KEYBOARD_HIGH
    white_count = 41
    white_width = (right - left) / white_count
    fall_window = 4.0
    pixels_per_second = (key_top - top) / fall_window
    draw.rounded_rectangle((left, top, right, bottom), radius=14, fill="#181a1f", outline="#444850", width=2)
    draw.text((left + 16, top + 14), "PIANO", fill="#d7d4ca", font=font)

    # Falling blocks reach the keyboard exactly on their note-on time.
    for note in notes:
        start, end = float(note.get("start", 0)), float(note.get("end", 0))
        pitch = int(note.get("pitch", 60))
        if pitch < low or pitch > high or start < time - 0.04 or start > time + fall_window:
            continue
        x, width, black = piano_key_geometry(pitch, left, white_width)
        y = key_top - (start - time) * pixels_per_second
        length = max(8, min(180, (end - start) * pixels_per_second))
        color = "#e9bc68" if black else "#d8ca8a"
        draw.rounded_rectangle((x + 2, y - length, x + width - 2, y), radius=4, fill=color, outline="#fff0bc")

    # White keys have the full natural-key height. Black keys are 62% height and narrower.
    for pitch in range(low, high + 1):
        if pitch % 12 not in WHITE_INDEX:
            continue
        x, width, _ = piano_key_geometry(pitch, left, white_width)
        attack = any(int(note.get("pitch", -1)) == pitch and abs(float(note.get("start", 0)) - time) < 1 / FPS
                     for note in notes)
        draw.rectangle((x, key_top, x + width, bottom), fill="#f7f5ee" if not attack else "#f0c76e", outline="#5b5b59")
    for pitch in range(low, high + 1):
        if pitch % 12 not in (1, 3, 6, 8, 10):
            continue
        x, width, _ = piano_key_geometry(pitch, left, white_width)
        attack = any(int(note.get("pitch", -1)) == pitch and abs(float(note.get("start", 0)) - time) < 1 / FPS
                     for note in notes)
        draw.rounded_rectangle((x, key_top, x + width, key_top + (bottom - key_top) * .62), radius=3,
                               fill="#d79c3f" if attack else "#161719", outline="#050506")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def chord_shape(name: str, shapes: dict) -> dict | None:
    """Use the same registered shape dictionary as the application."""
    base = name.split("/", 1)[0]
    return shapes.get(name) or shapes.get(base)


def chord_diagram(shape: dict | None, font) -> Image.Image | None:
    """Render a normal chord diagram, then rotate the complete image left."""
    if not shape:
        return None
    frets = shape.get("frets", [])
    if len(frets) != 6:
        return None
    # Render the ordinary upright diagram first so the requested rotation is exact.
    width, height, margin = 176, 128, 22
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = margin, 20, width - margin, height - 10
    string_step = (right - left) / 5
    fret_step = (bottom - top) / 5
    for string in range(6):
        x = left + string * string_step
        draw.line((x, top, x, bottom), fill="#e5dfce", width=2)
    for fret in range(6):
        y = top + fret * fret_step
        draw.line((left, y, right, y), fill="#e5dfce", width=3 if fret == 0 else 1)
    for string, fret in enumerate(frets):
        x = left + string * string_step
        if fret == -1:
            draw.text((x - 6, 1), "×", fill="#f49b9b", font=font)
        elif fret == 0:
            draw.ellipse((x - 4, 4, x + 4, 12), outline="#e5dfce", width=2)
        elif fret > 0:
            y = top + (fret - .5) * fret_step
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill="#f5c779")
    return image.transpose(Image.Transpose.ROTATE_90)


def render(folder: Path, audio: Path, output: Path, mode: str = "piano") -> None:
    import soundfile as sf
    duration = sf.info(str(audio)).duration
    grid = load(folder / "beat-grid.json")
    parts = {name: load(folder / f"{name}-midi.json") for name in ("vocals", "bass", "piano")}
    drums, chords = load(folder / "drums-midi.json"), load(folder / "acoustic-guitar-chords.json")
    shapes = load(Path(__file__).with_name("chord_shape.json"))
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        chord_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
    except OSError:
        font = chord_font = ImageFont.load_default()
    with tempfile.TemporaryDirectory(prefix="stem-json-video-") as temporary:
        frames = Path(temporary)
        for index in range(max(1, int(duration * FPS))):
            t = index / FPS
            image = Image.new("RGB", (WIDTH, HEIGHT), "#101114")
            draw = ImageDraw.Draw(image)
            title = "Piano MIDI" if mode == "piano" else "Acoustic guitar chords"
            draw.text((24, 16), f"{title}   {t:06.2f}s", fill="#ffffff", font=font)
            if mode == "piano":
                draw_falling_piano(draw, parts["piano"].get("notes", []), t, font)
            lanes = []
            for lane_index, (name, color) in enumerate(lanes):
                top, bottom = 58, 665
                draw.rectangle((24, top, WIDTH - 24, bottom), outline="#3e4147", fill="#1b1d22")
                draw.text((30, top + 6), name, fill="#d5d3cb", font=font)
                notes = parts[name].get("notes", [])
                for note in notes:
                    start, end = float(note.get("start", 0)), float(note.get("end", 0))
                    if end < t or start > t + WINDOW:
                        continue
                    x1 = 24 + max(0, start - t) / WINDOW * (WIDTH - 48)
                    x2 = 24 + min(WINDOW, end - t) / WINDOW * (WIDTH - 48)
                    pitch = int(note.get("pitch", 60))
                    y = bottom - 20 - (pitch % 36) / 35 * (bottom - top - 35)
                    draw.rounded_rectangle((x1, y - 5, max(x1 + 3, x2), y + 5), radius=3, fill=color)
            if mode == "chords":
                for beat in grid.get("beats", []):
                    beat = float(beat)
                    if t <= beat <= t + WINDOW:
                        x = 24 + (beat - t) / WINDOW * (WIDTH - 48)
                        draw.line((x, 48, x, 680), fill="#3a3d43")
            for chord in chords.get("chords", []):
                start, end = float(chord["start"]), float(chord["end"])
                if end < t or start > t + WINDOW:
                    continue
                # Do not clamp x1: old cards continue past the left edge and disappear.
                x1 = 24 + (start - t) / WINDOW * (WIDTH - 48)
                x2 = 24 + (end - t) / WINDOW * (WIDTH - 48)
                if mode == "piano":
                    continue
                right = max(x1 + 3, x2)
                draw.rounded_rectangle((x1 + 3, 62, right - 3, 668), radius=16,
                                       fill="#1e2025", outline="#565a61", width=2)
                draw.rectangle((x1 + 6, 66, right - 6, 72), fill="#c9a35f")
                label_box = draw.textbbox((0, 0), chord["chord"], font=chord_font)
                label_width = label_box[2] - label_box[0]
                draw.text(((x1 + right - label_width) / 2, 96), chord["chord"], fill="#fff4dc", font=chord_font)
                diagram = chord_diagram(chord_shape(chord["chord"], shapes), font)
                if diagram:
                    # 1.5× diagram, centered. Pillow clips it at the video edge while it flows.
                    diagram = diagram.resize((diagram.width * 3 // 2, diagram.height * 3 // 2), Image.Resampling.LANCZOS)
                    position = (int((x1 + right - diagram.width) / 2), 210)
                    image.paste(diagram, position, diagram)
            image.save(frames / f"frame_{index:06d}.png")
            if index % max(1, int(duration * FPS) // 100) == 0:
                print(f"EXPORT_PROGRESS {int((index + 1) * 70 / max(1, int(duration * FPS)))}", flush=True)
        print("EXPORT_PROGRESS 75", flush=True)
        subprocess.run(["ffmpeg", "-y", "-framerate", str(FPS), "-i", str(frames / "frame_%06d.png"), "-i", str(audio),
                        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-shortest", "-movflags", "+faststart", str(output)], check=True)
        print("EXPORT_PROGRESS 100", flush=True)


if __name__ == "__main__":
    render(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else "piano")
