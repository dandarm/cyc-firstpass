import numpy as np

from cyclone_locator.transforms.letterbox import (
    forward_map_xy,
    inverse_map_xy,
    resize_image,
)


def test_resize_stretch_round_trip():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    out, meta = resize_image(img, 224, mode="stretch")
    assert out.shape[:2] == (224, 224)
    assert meta["pad_x"] == 0
    assert meta["pad_y"] == 0
    assert meta["scale_x"] != meta["scale_y"]

    x_orig, y_orig = 123.4, 56.7
    x_lb, y_lb = forward_map_xy(x_orig, y_orig, meta)
    x_back, y_back = inverse_map_xy(x_lb, y_lb, meta)
    assert abs(x_back - x_orig) < 1e-6
    assert abs(y_back - y_orig) < 1e-6


def test_resize_letterbox_exposes_uniform_scale():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    out, meta = resize_image(img, 224, mode="letterbox")
    assert out.shape[:2] == (224, 224)
    assert abs(meta["scale_x"] - meta["scale_y"]) < 1e-12


def _run_without_pytest() -> int:
    try:
        test_resize_stretch_round_trip()
        test_resize_letterbox_exposes_uniform_scale()
    except AssertionError as exc:
        print(f"[FAIL] {exc}")
        return 1
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 2
    print("[OK] resize_image checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_without_pytest())
