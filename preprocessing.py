import pandas as pd
from typing import List, Tuple
import numpy as np


def extract_activity_segments(
        df_full: pd.DataFrame,
        df_annot: pd.DataFrame) -> Tuple[List[pd.DataFrame], List]:
    start = df_annot[["Start Timestamp (ms)",
                      "Stop Timestamp (ms)"]].to_numpy()
    start, end = start[:, 0], start[:, 1]
    out = []
    for s, e in zip(start, end):
        out.append(df_full[df_full["Timestamp (ms)"].between(
            s, e, inclusive="both")])
    labels = []
    for activity in df_annot['AnnotationId']:
        if 'rest' in activity.lower():
            labels.append(0)  # rest
        else:
            labels.append(1)  # stressful activity
    return out, labels


def get_intervals(df: pd.DataFrame, interval_duration: float = 1):
    time_attr = 'Timestamp (ms)'
    target_attr = 'Sample (V)'
    delta = df.iloc[1][time_attr] - df.iloc[0][time_attr]
    interval_length = int(60_000 * interval_duration / delta)
    arr = []

    epsilon = 1e-1
    num_intervals = int(df.shape[0] / interval_length + epsilon)

    # cramp is the additional length that needs to be
    # compensated for, i.e., even the alternate intervals
    # end up having some overlap
    cramp = num_intervals * interval_length - len(df)
    # if the size of interval is slightly bigger then the
    # overlap between consecutive intervals would
    # be a little less than 50%
    cramp = max(0, cramp)

    interval_indices = np.linspace(0,
                                   len(df) - cramp,
                                   num=2 * num_intervals,
                                   endpoint=False)[:-1]
    for i in interval_indices:
        i = int(i)
        interval = df[target_attr].iloc[i:i + interval_length].to_numpy()
        arr.append(interval)
    arr = np.array(arr)
    return arr
