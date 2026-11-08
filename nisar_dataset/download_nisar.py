"""Step 2b. Download NISAR GCOV files for one track/frame.

By default only dual-pol scenes (HH + HV) are downloaded. Add --all to also get the HH-only scenes.

Needs a free NASA Earthdata account (https://urs.earthdata.nasa.gov). Each file is ~1.5-2 GB.

Run (from the project folder), first with ONE file as a test:
    python nisar_dataset/download_nisar.py --track 154 --frame 90 --max 1
then every scene of that track/frame (HH-only ones included):
    python nisar_dataset/download_nisar.py --track 154 --frame 90 --all

Files go to data/nisar/ (this folder is NOT pushed to git).
"""
import argparse
from getpass import getpass
from pathlib import Path

import asf_search as asf
import pandas as pd

HERE = Path(__file__).parent              # scenes.csv lives next to this script
OUT = Path("data/nisar")

def main():
    """All code lives inside main(): on macOS, downloads run in child processes that re-import this file,
    and without this guard the script would start again (and ask for the password again) in each of them."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", type=int, required=True)
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--max", type=int, default=None, help="download only the N most recent scenes")
    ap.add_argument("--all", action="store_true", help="also download HH-only (single-pol) scenes")
    args = ap.parse_args()

    scenes = pd.read_csv(HERE / "scenes.csv", parse_dates=["date"])
    sel = scenes[(scenes.track == args.track) & (scenes.frame == args.frame)]
    if not args.all:
        sel = sel[sel.dual_pol]
    sel = sel.sort_values("date", ascending=False)
    if args.max:
        sel = sel.head(args.max)
    if sel.empty:
        raise SystemExit("No scenes for this track/frame. Check coverage.csv.")

    OUT.mkdir(parents=True, exist_ok=True)
    todo = [u for u in sel.url if not (OUT / u.rsplit("/", 1)[-1]).exists()]      # skip files already there
    print(f"{len(sel)} scenes selected, {len(todo)} to download, about {sel.size_gb.sum():.1f} GB in total.")
    print(sel[["date", "pol", "dual_pol", "size_gb"]].to_string(index=False))
    if todo:
        session = asf.ASFSession().auth_with_creds(input("Earthdata username: "), getpass("Earthdata password: "))
        asf.download_urls(todo, str(OUT), session=session, processes=1)
    print("Done. Files are in", OUT)


if __name__ == "__main__":
    main()