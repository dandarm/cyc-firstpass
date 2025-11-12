import pandas as pd

from cyclone_locator.datasets.windows_labeling import WindowsLabeling


def make_df():
    data = {
        "time": [
            "2021-01-01 10:00:00",
            "2021-01-01 10:30:00",
            "2021-01-01 11:00:00",
        ],
        "start_time": [
            "2021-01-01 10:00:00",
            "2021-01-01 10:00:00",
            "2021-01-01 10:00:00",
        ],
        "end_time": [
            "2021-01-01 11:00:00",
            "2021-01-01 11:00:00",
            "2021-01-01 11:00:00",
        ],
        "x_pix": [100, 110, 120],
        "y_pix": [200, 210, 220],
    }
    return pd.DataFrame(data)


def test_windows_inclusion_boundaries():
    df = make_df()
    labeler = WindowsLabeling.from_dataframe(df)
    assert labeler.is_positive(pd.Timestamp("2021-01-01 10:00:00"))
    assert labeler.is_positive(pd.Timestamp("2021-01-01 11:00:00"))
    assert not labeler.is_positive(pd.Timestamp("2021-01-01 09:59:00"))
    assert not labeler.is_positive(pd.Timestamp("2021-01-01 11:01:00"))


def test_windows_keypoint_lookup():
    df = make_df()
    labeler = WindowsLabeling.from_dataframe(df)
    kp = labeler.keypoint_for(pd.Timestamp("2021-01-01 10:30:00"))
    assert kp is not None
    assert kp.x == 110
    assert kp.y == 210
