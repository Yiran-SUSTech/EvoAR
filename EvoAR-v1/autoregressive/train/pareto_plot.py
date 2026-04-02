import math
from pathlib import Path

from PIL import Image, ImageDraw


BACKGROUND = (255, 255, 255)
AXIS = (40, 40, 40)
POINT = (49, 130, 206)
LINE = (220, 38, 38)
TEXT = (20, 20, 20)


def _normalize(values, lower, upper):
    if upper <= lower:
        return [0.5 for _ in values]
    return [(value - lower) / (upper - lower) for value in values]


def save_pareto_front_plot(archive, output_path, title=None, y_key="loss", y_label="loss", width=960, height=720):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)

    left = 90
    right = width - 40
    top = 50
    bottom = height - 90

    draw.line((left, bottom, right, bottom), fill=AXIS, width=2)
    draw.line((left, bottom, left, top), fill=AXIS, width=2)

    title = title or "Pareto Front"
    draw.text((left, 16), title, fill=TEXT)
    draw.text((right - 120, bottom + 28), "latency", fill=TEXT)
    draw.text((16, top - 4), y_label, fill=TEXT)

    if not archive:
        draw.text((left + 20, top + 20), "archive is empty", fill=TEXT)
        image.save(output_path)
        return

    points = sorted(
        [
            (float(item["latency"]), float(item[y_key]))
            for item in archive
            if y_key in item and math.isfinite(float(item["latency"])) and math.isfinite(float(item[y_key]))
        ],
        key=lambda point: (point[0], point[1]),
    )
    if not points:
        draw.text((left + 20, top + 20), f"archive has no finite points for {y_key}", fill=TEXT)
        image.save(output_path)
        return
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    x_pad = max((x_max - x_min) * 0.05, 1e-6)
    y_pad = max((y_max - y_min) * 0.05, 1e-6)
    x_norm = _normalize(xs, x_min - x_pad, x_max + x_pad)
    y_norm = _normalize(ys, y_min - y_pad, y_max + y_pad)

    pixel_points = []
    for x_value, y_value, x_ratio, y_ratio in zip(xs, ys, x_norm, y_norm):
        px = left + x_ratio * (right - left)
        py = bottom - y_ratio * (bottom - top)
        pixel_points.append((px, py, x_value, y_value))

    for tick_idx in range(5):
        ratio = tick_idx / 4
        tick_x = left + ratio * (right - left)
        tick_y = bottom - ratio * (bottom - top)
        draw.line((tick_x, bottom, tick_x, bottom + 6), fill=AXIS, width=1)
        draw.line((left - 6, tick_y, left, tick_y), fill=AXIS, width=1)
        x_value = (x_min - x_pad) + ratio * ((x_max + x_pad) - (x_min - x_pad))
        y_value = (y_min - y_pad) + ratio * ((y_max + y_pad) - (y_min - y_pad))
        draw.text((tick_x - 18, bottom + 10), f"{x_value:.3f}", fill=TEXT)
        draw.text((12, tick_y - 7), f"{y_value:.3f}", fill=TEXT)

    if len(pixel_points) >= 2:
        draw.line([(px, py) for px, py, _, _ in pixel_points], fill=LINE, width=2)

    radius = 5
    for px, py, _, _ in pixel_points:
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=POINT, outline=AXIS)

    summary = f"points={len(points)}  best_{y_key}={min(ys):.4f}  best_latency={min(xs):.4f}"
    draw.text((left, height - 36), summary, fill=TEXT)
    image.save(output_path)


def save_pareto_front_plots(archive, output_dir, step):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_pareto_front_plot(
        archive,
        output_dir / f"step_{step:07d}_loss_score.png",
        title=f"Pareto Front (loss score) @ step {step}",
        y_key="loss",
        y_label="loss score",
    )
    save_pareto_front_plot(
        archive,
        output_dir / f"step_{step:07d}_final_loss.png",
        title=f"Pareto Front (final loss) @ step {step}",
        y_key="final_loss",
        y_label="final loss",
    )
    save_pareto_front_plot(
        archive,
        output_dir / "latest_loss_score.png",
        title=f"Pareto Front (loss score) @ step {step}",
        y_key="loss",
        y_label="loss score",
    )
    save_pareto_front_plot(
        archive,
        output_dir / "latest_final_loss.png",
        title=f"Pareto Front (final loss) @ step {step}",
        y_key="final_loss",
        y_label="final loss",
    )
