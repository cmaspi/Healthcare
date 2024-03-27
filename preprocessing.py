from typing import List, Tuple

import numpy as np
import pandas as pd


def extract_activity_segments(
        df_full: pd.DataFrame,
        df_annot: pd.DataFrame) -> Tuple[List[pd.DataFrame], List]:
    start = df_annot[["Start Timestamp (ms)",
                      "Stop Timestamp (ms)"]].to_numpy()
    start, end = start[:, 0], start[:, 1]
    out = []
    labels = []

    for s, e, activity in zip(start, end, df_annot['AnnotationId']):
        if 'tsst' in activity.lower() and 'rest' in activity.lower():
            continue
        if 'cry' in activity.lower():
            continue
        if 'rest' in activity.lower():
            labels.append(0)  # rest
        else:
            labels.append(1)  # stressful activity
        out.append(df_full[df_full["Timestamp (ms)"].between(
            s, e, inclusive="both")])
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


def filter(df: pd.DataFrame):
    pass


def get_ema(df: pd.DataFrame):
    activities = df[~df['EventType'].isna()]['EventType'].unique()
    activities = list(activities)
    activities.remove('InfantCrying')
    ema_questions = [
        'Did you experience anything stressful?',
        'How stressed were you feeling?',
        'Did you feel you could not control important things?',
        'Did you feel confident in your ability to handle problems?',
        'Did you feel things are going your way?',
        'Did you feel difficulties piling up so you cannot overcome them?',
        'How happy were you feeling?', 'How excited were you feeling?',
        'How content were you feeling?', 'How worried were you feeling?',
        'How irritable/angry were you feeling?', 'How sad were you feeling?',
        'PSS', 'Induction'
    ]
    ema_responses = []
    for activity in activities:
        resp = df[df['EventType'] ==
                  activity].iloc[0][ema_questions].to_numpy()
        ema_responses.append(resp)
    ema_responses = np.array(ema_responses)
    return ema_responses, activities


def get_data(dir: str, interval_duration: float = 1):
    df_ema = pd.read_csv(f'{dir}/labeledfeatures.csv')
    df_annot = pd.read_csv(f'{dir}/annotations.csv')
    df_full = pd.read_csv(f'{dir}/elec.csv')

    ema_array, activities = get_ema(df_ema)
    out, labels = extract_activity_segments(df_full, df_annot)
    intervals = []
    for df, ema, label, act in zip(out, ema_array, labels, activities):
        arr = get_intervals(df, interval_duration)
        n = len(arr)
        arr = np.concatenate([
            arr, [ema] * n,
            np.array([act] * n)[:, None],
            np.array([label] * n)[:, None]
        ],
                             axis=1)
        intervals.append(arr)
    return np.concatenate(intervals, axis=0)
