from typing import Optional
from pathlib import Path
import polars as pl

from src.constants import PATH_TO_RESULTS

def loadDataframeCsvFile(pathToCsv: Path) -> Optional[pl.DataFrame]:
    try:
        return pl.read_csv(pathToCsv, null_values=["NA"])
    except Exception as e:
        print(f"Error loading dataframe from {pathToCsv}: {e}")
        return None

def getResultDataframe() -> Optional[pl.DataFrame]:
    return loadDataframeCsvFile(PATH_TO_RESULTS)