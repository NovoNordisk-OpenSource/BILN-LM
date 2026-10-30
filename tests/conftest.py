from pathlib import Path

import pandas as pd
import pytest

CORPUS = Path(__file__).resolve().parents[2] / "BILN-LM" / "data" / "biln_db.csv"


@pytest.fixture(scope="session")
def corpus() -> list[str]:
    """Real BILN strings from the original project, if that data is present."""
    if not CORPUS.is_file():
        pytest.skip(f"corpus not available: {CORPUS}")
    frame = pd.read_csv(CORPUS)
    return frame["biln"].dropna().astype(str).tolist()
