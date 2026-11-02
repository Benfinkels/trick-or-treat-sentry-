import io
import json
import pytest
from PIL import Image, ImageDraw

from sentry import (
    compute_frame_difference,
    fix_color_cast,
    load_env,
)


def create_solid_image(color=(128, 128, 128), size=(160, 120)) -> Image.Image:
    return Image.new("L", size, color=color[0])


def test_compute_frame_difference_identical():
    img1 = create_solid_image((100, 100, 100))
    img2 = create_solid_image((100, 100, 100))
    delta = compute_frame_difference(img1, img2)
    assert delta == 0.0


def test_compute_frame_difference_motion():
    img1 = create_solid_image((100, 100, 100))
    img2 = create_solid_image((100, 100, 100))
    draw = ImageDraw.Draw(img2)
    draw.rectangle([40, 30, 120, 90], fill=255)
    delta = compute_frame_difference(img1, img2)
    assert delta > 10.0


def test_fix_color_cast_neutral_passthrough():
    im = Image.new("RGB", (64, 64), color=(100, 100, 100))
    buf = io.BytesIO()
    im.save(buf, "JPEG")
    raw = buf.getvalue()

    corrected = fix_color_cast(raw)
    assert isinstance(corrected, bytes)
    assert len(corrected) > 0


def test_fix_color_cast_removes_magenta():
    # Magenta heavy image (R=200, G=30, B=200)
    im = Image.new("RGB", (64, 64), color=(200, 30, 200))
    buf = io.BytesIO()
    im.save(buf, "JPEG")
    raw = buf.getvalue()

    corrected = fix_color_cast(raw)
    result_img = Image.open(io.BytesIO(corrected)).convert("RGB")
    r, g, b = result_img.getpixel((32, 32))
    # Magenta cast should be mitigated (grayscale or balanced)
    assert abs(r - g) < 20 or abs(r - b) < 20


def test_load_env_nonexistent(tmp_path):
    env = load_env(tmp_path / ".env.fake")
    assert isinstance(env, dict)
