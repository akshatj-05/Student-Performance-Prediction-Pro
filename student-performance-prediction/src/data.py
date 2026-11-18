from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CSV_PATH = DATA_DIR / "student-por.csv"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"

# Earlier grades G1 and G2 are legitimate predictors only if they are available
# at the time the prediction is made. G3 is the value to predict.
NUMERIC_FEATURES = ["age", "studytime", "failures", "absences", "G1", "G2"]
CATEGORICAL_FEATURES = [
    "school", "sex", "address", "schoolsup", "famsup", "internet"
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "G3"


def load_dataset() -> pd.DataFrame:
    """Load the original UCI Portuguese-course CSV, downloading it once if needed."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        try:
            with urlopen(UCI_ZIP_URL, timeout=30) as response:
                archive_bytes = response.read()
            with ZipFile(BytesIO(archive_bytes)) as archive:
                member = next(
                    (name for name in archive.namelist() if name.endswith("student-por.csv")),
                    None,
                )
                if member is None:
                    raise FileNotFoundError("student-por.csv was not found in the UCI archive.")
                CSV_PATH.write_bytes(archive.read(member))
        except Exception as exc:
            raise RuntimeError(
                "Could not download the UCI dataset. Check your internet connection, "
                "then try again. You can also download student-por.csv from the UCI "
                "Student Performance dataset page and place it in the project's data/ folder."
            ) from exc

    df = pd.read_csv(CSV_PATH, sep=";")
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    return df[FEATURES + [TARGET]].dropna().copy()
