"""Stream GH Archive hour files and keep only the cohort users' events.

preprocess.py (official) only keeps lines whose actor is a cohort user (Step 2). Doing the
same filter while downloading means we never store the full ~15 MB/hour files: each kept
hour file is a few KB, and running the official preprocess.py on the filtered folder gives
exactly the same tables as running it on the raw files.

Usage:
    python src/download_filter.py --start 2026-05-01 --end 2026-11-18T10 --out raw_filtered
"""
import argparse
import gzip
import os
import sys
import time
import urllib.error
import urllib.request
import zlib
from datetime import datetime, timedelta, timezone
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "assignment2" / "scripts"))
from preprocess import ACTOR_ID_RE, code_int, load_cohort  # noqa: E402

COHORT = None
OUT = None


def _init(out):
    global COHORT, OUT
    COHORT = load_cohort(ROOT / "assignment2" / "scripts" / "cohort.txt")
    OUT = Path(out)


def fetch(h):
    name = f"{h:%Y-%m-%d}-{h.hour}.json.gz"
    dst = OUT / name
    if dst.exists():
        return name, "skip", 0
    url = f"https://data.gharchive.org/{name}"
    for attempt in range(4):
        try:
            kept, checked = [], {}
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                with gzip.GzipFile(fileobj=resp) as gz:
                    try:
                        for line in gz:
                            m = ACTOR_ID_RE.search(line)
                            if not m:
                                continue
                            a = int(m.group(1))
                            if a not in checked:
                                checked[a] = code_int("user", a) in COHORT
                            if checked[a]:
                                kept.append(line.rstrip(b"\n"))
                    except (EOFError, gzip.BadGzipFile, zlib.error):
                        pass  # damaged end: keep what we read, same as preprocess.py
            tmp = dst.with_suffix(".tmp")
            with gzip.open(tmp, "wb") as f:
                f.write(b"\n".join(kept) + (b"\n" if kept else b""))
            os.replace(tmp, dst)
            return name, "ok", len(kept)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return name, "missing", 0
            time.sleep(3 * (attempt + 1))
        except Exception:  # network hiccup
            time.sleep(3 * (attempt + 1))
    return name, "fail", 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True, help="last hour (inclusive), e.g. 2026-11-18T10")
    ap.add_argument("--out", default=str(ROOT / "raw_filtered"))
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    s = datetime.fromisoformat(a.start).replace(tzinfo=timezone.utc)
    e = datetime.fromisoformat(a.end).replace(tzinfo=timezone.utc)
    hours = []
    while s <= e:
        hours.append(s)
        s += timedelta(hours=1)
    stats = {}
    log = open(Path(a.out) / "_download_log.txt", "a")
    with Pool(a.workers, initializer=_init, initargs=(a.out,)) as pool:
        for i, (name, st, n) in enumerate(pool.imap_unordered(fetch, hours), 1):
            stats[st] = stats.get(st, 0) + 1
            if st in ("missing", "fail"):
                log.write(f"{name}\t{st}\n"); log.flush()
            if i % 48 == 0 or i == len(hours):
                print(f"{i}/{len(hours)} {stats}", flush=True)
    print("done", stats)


if __name__ == "__main__":
    main()
