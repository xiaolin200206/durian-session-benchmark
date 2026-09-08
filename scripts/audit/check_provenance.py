#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_provenance.py
===================
Checks whether orchard-level provenance can be recovered for the durian dataset.

Just run it. No arguments needed. The path is already set below.

    python check_provenance.py

It prints everything to the screen AND writes provenance_report.txt into the
zenodo folder, so you can just send me that one file.

What it checks
--------------
1. sessions.xlsx / sessions.csv  -- is there already a site/orchard column?
2. EXIF in the originals -- did GPS, capture timestamps, or device tags
   survive the resizing and format conversion?
3. If GPS survived: clusters coordinates into orchards automatically and
   writes a per-image orchard assignment file.
4. If only timestamps survived: groups by capture date as a "visit" proxy.

Reads images directly out of the .zip files, so nothing needs extracting.
"""

import os
import sys
import io
import zipfile
import collections
import subprocess
from pathlib import Path
from datetime import datetime

# =========================================================================
# THE PATH -- change this line only if you move the folder
# =========================================================================
import argparse as _ap
_p=_ap.ArgumentParser(add_help=True)
_p.add_argument("--data", default=".",
                help="folder holding sessions.csv, splits/ and the image dirs")
BASE = Path(_p.parse_known_args()[0].data)
# =========================================================================

# Cap on how many images to scan. Set to None to scan everything.
MAX_IMAGES = None


# ---------------------------------------------------------------- plumbing
_report_lines = []


def say(msg=""):
    print(msg)
    _report_lines.append(str(msg))


def ensure(pkg, import_name=None):
    """Install a package if missing, so the script stays double-clickable."""
    name = import_name or pkg
    try:
        __import__(name)
        return True
    except ImportError:
        say(f"  (installing {pkg} ...)")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"],
                           check=True)
            __import__(name)
            return True
        except Exception as e:
            say(f"  !! could not install {pkg}: {e}")
            return False


# ------------------------------------------------------------- 1. tabular
LOCATION_HINTS = ['site', 'orchard', 'location', 'farm', 'plot', 'block',
                  'gps', 'lat', 'lon', 'address', 'district', 'estate',
                  'place', 'kebun', 'ladang', 'tempat', 'source']


def _scan_header(header, rows):
    lower = [str(h).lower() for h in header]
    hits = [(i, h) for i, h in enumerate(lower)
            if any(hint in h for hint in LOCATION_HINTS)]
    if hits:
        say(f"  >>> POSSIBLE SITE/ORCHARD COLUMN(S): {[h for _, h in hits]}")
        for i, h in hits:
            vals = collections.Counter(
                str(r[i]) for r in rows if i < len(r) and r[i] not in (None, ""))
            say(f"      '{header[i]}' -> {len(vals)} distinct values")
            for v, n in vals.most_common(20):
                say(f"          {v!r}: {n}")
    else:
        say("  >>> No column name suggests site/orchard/location.")
        say("      (Check the column list above by eye in case it is named "
            "something unexpected.)")


def check_table():
    say("\n" + "=" * 72)
    say("1. SESSIONS TABLE -- is orchard/site already recorded?")
    say("=" * 72)

    found_any = False
    for path in (BASE / "sessions.xlsx", BASE / "sessions.csv"):
        if not path.exists():
            continue
        found_any = True
        say(f"\nFile: {path.name}")
        try:
            if path.suffix.lower() == ".xlsx":
                if not ensure("openpyxl"):
                    continue
                import openpyxl
                wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
                for sheet_name in wb.sheetnames:
                    all_rows = list(wb[sheet_name].iter_rows(values_only=True))
                    if not all_rows:
                        continue
                    header = [str(h) if h is not None else "" for h in all_rows[0]]
                    rows = all_rows[1:]
                    say(f"  Sheet '{sheet_name}': {len(rows)} data rows")
                    say(f"  Columns: {header}")
                    _scan_header(header, rows)
            else:
                import csv
                with open(path, newline='', encoding='utf-8-sig') as f:
                    r = list(csv.reader(f))
                if r:
                    say(f"  {len(r) - 1} data rows")
                    say(f"  Columns: {r[0]}")
                    _scan_header(r[0], r[1:])
        except Exception as e:
            say(f"  !! could not read {path.name}: {e}")

    if not found_any:
        say("\n  Neither sessions.xlsx nor sessions.csv found in that folder.")


# ---------------------------------------------------------------- 2. EXIF
EXTS = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.tif', '.tiff', '.webp'}


def iter_images():
    """Yield (label, bytes) from loose folders and from zips, without extracting."""
    for sub in ("images_fullres", "images_512"):
        d = BASE / sub
        if d.is_dir():
            for p in sorted(d.rglob("*")):
                if p.suffix.lower() in EXTS:
                    try:
                        yield f"{sub}/{p.relative_to(d).as_posix()}", p.read_bytes()
                    except Exception:
                        pass

    for sub in ("images_fullres.zip", "images_512.zip"):
        z = BASE / sub
        if z.is_file():
            try:
                with zipfile.ZipFile(z) as zf:
                    for name in zf.namelist():
                        if Path(name).suffix.lower() in EXTS:
                            try:
                                yield f"{sub}:{name}", zf.read(name)
                            except Exception:
                                pass
            except Exception as e:
                say(f"  !! could not open {sub}: {e}")


def to_degrees(value):
    def f(x):
        try:
            return float(x)
        except Exception:
            return float(x[0]) / float(x[1])
    d, m, s = value
    return f(d) + f(m) / 60.0 + f(s) / 3600.0


def check_exif():
    say("\n" + "=" * 72)
    say("2. EXIF -- did GPS / timestamp / device survive?")
    say("=" * 72)

    if not ensure("pillow", "PIL"):
        say("  Cannot proceed without Pillow.")
        return [], []

    from PIL import Image
    from PIL.ExifTags import TAGS, GPSTAGS

    heic_ok = ensure("pillow-heif", "pillow_heif")
    if heic_ok:
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except Exception:
            heic_ok = False

    n_total = n_exif = n_gps = n_dt = n_dev = failed = 0
    devices = collections.Counter()
    coords, stamps = [], []

    for label, blob in iter_images():
        if MAX_IMAGES is not None and n_total >= MAX_IMAGES:
            break
        n_total += 1
        if n_total % 100 == 0:
            print(f"    ... scanned {n_total} images", end="\r")

        merged, ex = {}, None
        try:
            img = Image.open(io.BytesIO(blob))
            ex = img.getexif()
            if ex:
                for k, v in ex.items():
                    merged[TAGS.get(k, k)] = v
                try:
                    for k, v in ex.get_ifd(0x8769).items():   # Exif sub-IFD
                        merged[TAGS.get(k, k)] = v
                except Exception:
                    pass
        except Exception:
            failed += 1
            continue

        if not merged:
            continue
        n_exif += 1

        if merged.get('Make') or merged.get('Model'):
            n_dev += 1
            devices[(str(merged.get('Make', '?')).strip('\x00 '),
                     str(merged.get('Model', '?')).strip('\x00 '))] += 1

        dt = merged.get('DateTimeOriginal') or merged.get('DateTime')
        if dt:
            n_dt += 1
            stamps.append((label, str(dt)))

        try:
            gps_ifd = ex.get_ifd(0x8825) if ex else None
        except Exception:
            gps_ifd = None
        if gps_ifd:
            g = {GPSTAGS.get(k, k): v for k, v in gps_ifd.items()}
            if 'GPSLatitude' in g and 'GPSLongitude' in g:
                try:
                    lat = to_degrees(g['GPSLatitude'])
                    if str(g.get('GPSLatitudeRef', 'N')).upper().startswith('S'):
                        lat = -lat
                    lon = to_degrees(g['GPSLongitude'])
                    if str(g.get('GPSLongitudeRef', 'E')).upper().startswith('W'):
                        lon = -lon
                    coords.append((label, lat, lon))
                    n_gps += 1
                except Exception:
                    pass

    print(" " * 60, end="\r")
    say(f"\nImages scanned:              {n_total}")
    if failed:
        note = "  -- may be HEIC; pillow-heif unavailable" if not heic_ok else ""
        say(f"  (could not open:           {failed}{note})")
    say(f"With any EXIF block:         {n_exif}")
    say(f"With Make/Model:             {n_dev}")
    for (mk, md), n in devices.most_common(10):
        say(f"    {mk} {md}: {n}")
    say(f"With DateTime:               {n_dt}")
    for label, dt in stamps[:5]:
        say(f"    {label}: {dt}")
    say(f"With usable GPS:             {n_gps}")
    for label, lat, lon in coords[:5]:
        say(f"    {label}: {lat:.6f}, {lon:.6f}")

    return coords, stamps


# ------------------------------------------------------- 3. cluster by GPS
def cluster_gps(coords):
    say("\n" + "=" * 72)
    say("3. ORCHARD RECOVERY from GPS")
    say("=" * 72)
    say(f"{len(coords)} images carry coordinates. Clustering into orchards.")

    import math
    RADIUS_M = 300.0

    def haversine(a, b):
        R = 6371000.0
        la1, lo1, la2, lo2 = map(math.radians, [a[0], a[1], b[0], b[1]])
        dla, dlo = la2 - la1, lo2 - lo1
        h = math.sin(dla / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlo / 2) ** 2
        return 2 * R * math.asin(math.sqrt(h))

    clusters = []
    for label, lat, lon in coords:
        placed = False
        for c in clusters:
            if haversine((lat, lon), c['centre']) <= RADIUS_M:
                c['members'].append(label)
                n = len(c['members'])
                c['centre'] = ((c['centre'][0] * (n - 1) + lat) / n,
                               (c['centre'][1] * (n - 1) + lon) / n)
                placed = True
                break
        if not placed:
            clusters.append({'centre': (lat, lon), 'members': [label]})

    ordered = sorted(clusters, key=lambda c: -len(c['members']))
    say(f"\n>>> {len(ordered)} distinct orchard locations "
        f"(clustering radius {RADIUS_M:.0f} m):\n")
    for i, c in enumerate(ordered, 1):
        say(f"  Orchard {i:>2}: {len(c['members']):>4} images   "
            f"centre {c['centre'][0]:.5f}, {c['centre'][1]:.5f}")

    try:
        import csv as _csv
        idx = {}
        for i, c in enumerate(ordered, 1):
            for m in c['members']:
                idx[m] = i
        out = BASE / "image_orchard_map.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["image", "orchard_id", "lat", "lon"])
            for label, lat, lon in coords:
                w.writerow([label, idx.get(label, ""), f"{lat:.6f}", f"{lon:.6f}"])
        say(f"\nWrote per-image orchard assignments to: {out.name}")
    except Exception as e:
        say(f"  !! could not write image_orchard_map.csv: {e}")


# --------------------------------------------------- 4. fallback by date
def cluster_dates(stamps):
    say("\n" + "=" * 72)
    say("4. FALLBACK -- grouping by capture date (a 'visit' proxy)")
    say("=" * 72)
    by_date = collections.defaultdict(list)
    for label, dt in stamps:
        by_date[str(dt).split(" ")[0]].append(label)
    say(f"{len(by_date)} distinct capture dates across {len(stamps)} images:\n")
    for d, names in sorted(by_date.items()):
        say(f"  {d}: {len(names):>4} images")
    say("\nEach date is roughly one field visit. Two visits to the same orchard")
    say("appear as two dates, so this over-splits -- but it is still a stricter")
    say("grouping than capture session.")


# -------------------------------------------------------------------- main
def main():
    say("PROVENANCE CHECK")
    say(f"Run at: {datetime.now():%Y-%m-%d %H:%M:%S}")
    say(f"Folder: {BASE}")

    if not BASE.exists():
        say("\n!! That folder does not exist. Check the BASE path at the top of this file.")
        return

    say(f"\nContents: {sorted(p.name for p in BASE.iterdir())}")

    check_table()
    coords, stamps = check_exif()

    if coords:
        cluster_gps(coords)
    else:
        say("\n" + "=" * 72)
        say("3. ORCHARD RECOVERY from GPS")
        say("=" * 72)
        say("No GPS coordinates survived. Orchard-level grouping cannot be")
        say("recovered this way.")

    if stamps:
        cluster_dates(stamps)

    say("\n" + "=" * 72)
    say("VERDICT")
    say("=" * 72)
    if coords:
        say("GPS survived -> orchard-level grouping IS recoverable.")
        say("Send image_orchard_map.csv and the leave-one-orchard-out analysis")
        say("plus the four-rung sampling ladder can be added to the paper.")
    elif stamps:
        say("No GPS, but timestamps survived -> a date-based 'visit' grouping is")
        say("possible. Weaker than orchard, stronger than capture session.")
    else:
        say("No GPS and no timestamps -> nothing recoverable. The manuscript's")
        say("wording in Sect. 3.4 and 5.1 is correct as written, and this report")
        say("is the evidence for it. Nothing needs to change.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        say("\n!! unexpected error:")
        say(traceback.format_exc())
    finally:
        try:
            out = BASE / "provenance_report.txt"
            out.write_text("\n".join(_report_lines), encoding="utf-8")
            print(f"\n\nReport saved to: {out}")
        except Exception as e:
            print(f"\n(could not save report file: {e})")
