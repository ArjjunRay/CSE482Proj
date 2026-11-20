from pathlib import Path

import pandas as pd
import pytest

from nascar_performance.demo import generate_demo_data


def test_generate_demo_data_is_reproducible(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    assert generate_demo_data(first) == generate_demo_data(second)
    pd.testing.assert_frame_equal(
        pd.read_csv(first / "race_results.csv"), pd.read_csv(second / "race_results.csv")
    )


def test_generate_demo_data_rejects_tiny_dataset(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least"):
        generate_demo_data(tmp_path, race_count=2, driver_count=2)
