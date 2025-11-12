import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

try:
    from cyclone_locator.datasets.med_fullbasin import MedFullBasinDataset
except ImportError as exc:  # pragma: no cover - dipende da runtime
    pytest.skip(f"MedFullBasinDataset import skipped: {exc}", allow_module_level=True)


def test_manifest_cli_smoke(tmp_path):
    out_dir = tmp_path / "manifests"
    windows_csv = Path("mini_data_input/medicanes_new_windows.csv").resolve()
    images_dir = Path("mini_data_input/resized").resolve()

    cmd = [
        sys.executable,
        str(Path("scripts/make_manifest_from_windows.py").resolve()),
        "--windows-csv",
        str(windows_csv),
        "--images-dir",
        str(images_dir),
        "--out-dir",
        str(out_dir),
        "--orig-size",
        "1290",
        "420",
        "--target-size",
        "512",
    ]
    subprocess.run(cmd, check=True)

    for split in ("train", "val", "test"):
        assert (out_dir / f"{split}.csv").exists()

    df = pd.read_csv(out_dir / "train.csv")
    assert "presence" in df.columns
    assert "image_path" in df.columns

    ds = MedFullBasinDataset(
        out_dir / "train.csv",
        image_size=512,
        heatmap_stride=4,
        heatmap_sigma_px=8,
        use_aug=False,
        use_pre_letterboxed=False,
    )
    sample = ds[0]
    assert sample["image"].shape[-1] == 512
    assert sample["presence"].shape == (1,)
