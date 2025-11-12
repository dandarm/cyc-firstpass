from cyclone_locator.datasets.windows_labeling import (
    compute_letterbox_params,
    project_keypoint,
    unproject_keypoint,
)


def test_letterbox_origin_maps_to_padding():
    params = compute_letterbox_params(1290, 420, 512)
    x_lb, y_lb = project_keypoint(0, 0, params)
    assert abs(x_lb - params.pad_x) <= 1
    assert abs(y_lb - params.pad_y) <= 1


def test_letterbox_round_trip_precision():
    params = compute_letterbox_params(1290, 420, 512)
    x_lb, y_lb = project_keypoint(320.5, 111.5, params)
    x_orig, y_orig = unproject_keypoint(x_lb, y_lb, params)
    assert abs(x_orig - 320.5) <= 1.0
    assert abs(y_orig - 111.5) <= 1.0
