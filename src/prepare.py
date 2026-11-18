"""Stage 1: build point-in-time features and labels for every weekly prediction time.

Prediction times t are aligned with the test week (Saturday 16:00 UTC = Sunday 00:00 Taiwan).
Features use cutoff = t - GAP, where GAP = (test week start) - (end of our data), so every
training sample sees exactly the same "blind" period as the real test prediction.

Output: work/dataset.pkl
"""
import pickle

import numpy as np
import pandas as pd

from common import (TARGETS, TEST_T, WEEK, WORK, activity_log, data_end, labels_for,
                    load_tables, ROOT)
from features import FeatureBuilder

MIN_HISTORY = pd.Timedelta(days=28)
DATA_START = pd.Timestamp("2025-12-01")


def main():
    tb = load_tables()
    log = activity_log(tb)
    end = data_end(tb)
    gap = (TEST_T - end - pd.Timedelta(seconds=1)).floor("h")
    gap = max(gap, pd.Timedelta(0))
    print(f"data end {end}, test start {TEST_T} UTC  ->  gap {gap}")

    fb = FeatureBuilder(tb, log)
    cohort = fb.cohort
    # labelled prediction times: label week fully inside the data
    ts, t = [], TEST_T - WEEK
    while t - gap - MIN_HISTORY >= DATA_START:
        if t + WEEK <= end:
            ts.append(t)
        t -= WEEK
    ts = sorted(ts)
    print(f"{len(ts)} labelled weeks: {ts[0]} ... {ts[-1]}")

    samples = {}
    for t in ts:
        X = fb.build(t - gap)
        y = labels_for(log, cohort, t)
        samples[t] = (X, y)
        print(t, X.shape, "pos rate", y.mean(0).round(3))
    X_test = fb.build(TEST_T - gap)

    users = tb["Users"]
    gh = users.set_index("user_id").github_user_id
    test_ids = pd.read_csv(ROOT / "assignment2" / "submission" / "test_input.csv").UserID
    in_data = test_ids.isin(gh[cohort].values)
    print(f"test users: {len(test_ids)}, found in data: {in_data.sum()}")

    with open(WORK / "dataset.pkl", "wb") as f:
        pickle.dump({"samples": samples, "X_test": X_test, "cohort": cohort,
                     "gh_of_cohort": gh[cohort].to_numpy(), "test_ids": test_ids.to_numpy(),
                     "gap": gap, "end": end, "targets": TARGETS}, f)
    log.to_parquet(WORK / "activity_log.parquet")


if __name__ == "__main__":
    main()
