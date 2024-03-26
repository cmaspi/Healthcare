import pandas as pd
from typing import List


def extract_activity_segments(df_full: pd.DataFrame,
                              df_annot: pd.DataFrame) -> List[pd.DataFrame]:
    start = df_annot[["Start Timestamp (ms)",
                      "Stop Timestamp (ms)"]].to_numpy()
    start, end = start[:, 0], start[:, 1]
    out = []
    for s, e in zip(start, end):
        out.append(df_full[df_full["Timestamp (ms)"].between(
            s, e, inclusive="both")])
    return out
