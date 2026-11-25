r"""
make_sample.py - OPTIONAL quick dry run.
Creates a small stratified sample of CICIDS2017 so you can test notebook 01 in about 2 minutes.

    python make_sample.py                      # reads data/cicids, writes data/cicids_sample/sample.csv

Then run the notebook against it:
    Windows (cmd):   set CICIDS_DIR=data\cicids_sample   then start jupyter from the same window
    Mac/Linux:       CICIDS_DIR=data/cicids_sample jupyter notebook
"""
import glob
import sys
from pathlib import Path

import pandas as pd

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/cicids")
DST = Path("data/cicids_sample")
CAP = 8000                      # max rows per label per file

DST.mkdir(parents=True, exist_ok=True)
parts = []
for f in sorted(glob.glob(str(SRC / "*.csv"))):
    df = pd.read_csv(f, low_memory=False, encoding="latin1")
    df.columns = df.columns.str.strip()
    # loop over groups (pandas groupby.apply can silently drop the group column)
    parts.append(pd.concat([g.sample(min(len(g), CAP), random_state=1)
                            for _, g in df.groupby("Label")], ignore_index=True))
    print("read", Path(f).name)
sample = pd.concat(parts, ignore_index=True)
sample.to_csv(DST / "sample.csv", index=False)
print(sample.shape)
print(sample["Label"].value_counts())
