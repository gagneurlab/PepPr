#!/usr/bin/env python3
"""
Download PRIDE submissions for Müller et al. Nature 2020
("The proteome landscape of the kingdoms of life"):
  PXD014877 — main study (raw + MaxQuant Search.zip + FASTA archive)
  PXD019483 — companion / validation runs (raw + search result tables)

Uses the PRIDE Archive v2 files API and wget(1) with resume over HTTPS.

Müller's raws use the German common name in the filename (HeLa for human,
Maus for mouse, …), not the Latin species. Pass ``--species <key>`` (where
<key> is one of the entries in ``KOL_SPECIES_RAW_PATTERNS``, e.g. ``human``
or ``mouse``) to download only that species' raws and to auto-extract the
matching per-species ``msms.txt`` from a locally-present ``Search.zip``
into ``PXD014877/search_results/<SearchDir>/msms.txt``.

The bulk MaxQuant output for PXD014877 is bundled as ``Search.zip`` (31 GB,
contains per-species ``msms.txt``/``evidence.txt``/etc. for ~100 organisms).
Use ``--with-search-zip`` to add it to the download set even when a name
filter would otherwise exclude it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from typing import Any, Pattern

# Make the dnps_hybrid package importable when this file is invoked as a
# script from the project root (`python scripts/pride_download_...`).
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from dnps_hybrid.const import KOL_SPECIES_DIRS, KOL_SPECIES_RAW_PATTERNS  # noqa: E402


PRIDE_FILES_API = "https://www.ebi.ac.uk/pride/ws/archive/v2/projects/{pxd}/files"


def ftp_to_https(url: str) -> str:
    if url.startswith("ftp://ftp.pride.ebi.ac.uk"):
        return "https://ftp.pride.ebi.ac.uk" + url[len("ftp://ftp.pride.ebi.ac.uk") :]
    return url


def pick_download_url(file_record: dict[str, Any]) -> str | None:
    locs = file_record.get("publicFileLocations") or []
    https = None
    ftp = None
    for loc in locs:
        if loc.get("name") == "FTP Protocol":
            v = loc.get("value")
            if not v:
                continue
            h = ftp_to_https(v)
            if h.startswith("https://"):
                https = h
            ftp = v
    return https or ftp


def fetch_file_list(pxd: str, page_size: int = 100) -> list[dict[str, Any]]:
    page = 0
    out: list[dict[str, Any]] = []
    while True:
        url = f"{PRIDE_FILES_API.format(pxd=pxd)}?pageSize={page_size}&page={page}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            batch = json.loads(resp.read().decode())
        if not batch:
            break
        out.extend(batch)
        if len(batch) < page_size:
            break
        page += 1
    return out


def category_of(rec: dict[str, Any]) -> str:
    cat = (rec.get("fileCategory") or {}).get("value") or "UNKNOWN"
    return str(cat)


def want_file(rec: dict[str, Any], include: set[str]) -> bool:
    cat = category_of(rec)
    if cat in include:
        return True
    return False


def compile_name_filter(
    species: str | None,
    human_only: bool,
    name_regex: str | None,
) -> Pattern[str] | None:
    if name_regex:
        return re.compile(name_regex)
    if species:
        return re.compile(KOL_SPECIES_RAW_PATTERNS[species], re.IGNORECASE)
    if human_only:
        # Back-compat: --human-only is equivalent to --species human.
        return re.compile(KOL_SPECIES_RAW_PATTERNS["human"], re.IGNORECASE)
    return None


def extract_species_msms(
    search_zip_path: str,
    species: str,
    out_search_dir: str,
) -> bool:
    """If ``Search.zip`` is present locally, extract that species' msms.txt
    (and a couple of small companion tables) into
    ``out_search_dir/<SearchDir>/``. Returns True if anything was extracted.

    This is a no-op when ``Search.zip`` is missing — useful for users who
    don't want the 31 GB bundle.
    """
    if not os.path.isfile(search_zip_path):
        print(f"  (no local Search.zip at {search_zip_path}; skip msms extraction)")
        return False
    sub = KOL_SPECIES_DIRS[species]
    wanted = {f"{sub}/msms.txt", f"{sub}/parameters.txt", f"{sub}/summary.txt"}
    out_dir = os.path.join(out_search_dir, sub)
    os.makedirs(out_dir, exist_ok=True)
    extracted = []
    with zipfile.ZipFile(search_zip_path) as zf:
        names = set(zf.namelist())
        for member in wanted:
            if member not in names:
                continue
            target = os.path.join(out_dir, os.path.basename(member))
            if os.path.exists(target):
                print(f"  EXISTS skip: {target}")
                extracted.append(target)
                continue
            print(f"  extracting Search.zip:{member} -> {target}")
            with zf.open(member) as src, open(target, "wb") as dst:
                while True:
                    chunk = src.read(8 << 20)
                    if not chunk:
                        break
                    dst.write(chunk)
            extracted.append(target)
    return bool(extracted)


def file_matches_filter(
    name: str,
    pattern: Pattern[str] | None,
    also_substrings: list[str],
) -> bool:
    if pattern is not None and pattern.search(name):
        return True
    for sub in also_substrings:
        if sub and sub in name:
            return True
    return False


def run_wget(url: str, out_path: str, wget_bin: str) -> int:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    partial = out_path + ".wget-tmp"
    cmd = [
        wget_bin,
        "--continue",
        "--timeout=60",
        "--tries=0",
        "--read-timeout=120",
        "--waitretry=30",
        # -nv: suppress per-50K dotted progress bar; emit one line per file
        # (URL -> path [size] saved) plus errors. Crucial when running under
        # slurm where stdout/stderr is teed to a file — the default progress
        # output ballooned the 19047882 log to 500k+ lines.
        "-nv",
        "-O",
        partial,
        url,
    ]
    p = subprocess.run(cmd)
    if p.returncode == 0 and os.path.exists(partial):
        os.replace(partial, out_path)
    return p.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description="Download Kingdoms of Life PRIDE data.")
    ap.add_argument(
        "--dest",
        default=os.environ.get(
            "KOL_DEST",
            "/s/project/denovo-prosit/SamKhan/kingdoms_of_life",
        ),
        help="Root output directory (default: $KOL_DEST or "
        "/s/project/denovo-prosit/SamKhan/kingdoms_of_life — matches "
        "dnps_hybrid.const.KOL_DATA_ROOT).",
    )
    ap.add_argument(
        "--pxd",
        action="append",
        default=None,
        help="PRIDE accession (repeatable). Default: PXD014877 PXD019483",
    )
    ap.add_argument(
        "--include",
        default="RAW,SEARCH,OTHER",
        help="Comma-separated PRIDE file categories to download (RAW,SEARCH,OTHER).",
    )
    ap.add_argument(
        "--wget",
        default=os.environ.get("WGET", "wget"),
        help="wget binary path.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="List planned downloads only.",
    )
    ap.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Stop after N successful downloads (0 = no limit). For smoke tests.",
    )
    ap.add_argument(
        "--species",
        default=None,
        choices=sorted(KOL_SPECIES_RAW_PATTERNS),
        help="Restrict downloads to one species (filename regex from "
        "KOL_SPECIES_RAW_PATTERNS) and auto-extract that species' msms.txt "
        "from a local Search.zip into search_results/<SearchDir>/.",
    )
    ap.add_argument(
        "--human-only",
        action="store_true",
        help="Deprecated alias for --species human.",
    )
    ap.add_argument(
        "--name-regex",
        default=None,
        metavar="PATTERN",
        help="Only download files whose fileName matches this regex "
        "(overrides --species/--human-only).",
    )
    ap.add_argument(
        "--also-substring",
        action="append",
        default=[],
        metavar="SUB",
        help="Also download any file whose fileName contains SUB (repeatable). "
        "Use e.g. --also-substring peptides.txt for PXD019483 tables.",
    )
    ap.add_argument(
        "--with-search-zip",
        action="store_true",
        help="Also download PXD014877's full MaxQuant Search.zip (~31 GB), which "
        "contains per-species msms/evidence/peptides tables for all organisms in "
        "the KoL study. Bypasses --human-only/--name-regex filters for that file.",
    )
    args = ap.parse_args()

    pxds = args.pxd or ["PXD014877", "PXD019483"]
    include = {x.strip().upper() for x in args.include.split(",") if x.strip()}
    name_pat = compile_name_filter(args.species, args.human_only, args.name_regex)
    also_subs: list[str] = list(args.also_substring)
    if args.with_search_zip and "Search.zip" not in also_subs:
        also_subs.append("Search.zip")
    if name_pat is None and not also_subs:
        filter_desc = "(none — all files in selected categories)"
    else:
        regex_disp = (
            args.name_regex
            or (KOL_SPECIES_RAW_PATTERNS[args.species] if args.species else None)
            or (KOL_SPECIES_RAW_PATTERNS["human"] if args.human_only else None)
        )
        filter_desc = f"regex={regex_disp!r} also_substrings={also_subs!r}"

    dest_root = os.path.abspath(args.dest)
    print(f"Destination: {dest_root}", flush=True)
    print(f"Projects: {pxds}", flush=True)
    print(f"Categories: {sorted(include)}", flush=True)
    print(f"Name filter: {filter_desc}", flush=True)

    done = 0
    max_files = args.max_files

    for pxd in pxds:
        print(f"\n=== Listing {pxd} ===", flush=True)
        try:
            files = fetch_file_list(pxd)
        except urllib.error.URLError as e:
            print(f"ERROR listing {pxd}: {e}", file=sys.stderr)
            return 1
        print(f"API returned {len(files)} files.", flush=True)

        subdirs = {"RAW": "raw", "SEARCH": "search_results", "OTHER": "other"}
        planned = [f for f in files if want_file(f, include)]
        if name_pat is not None or also_subs:
            before = len(planned)
            planned = [
                f
                for f in planned
                if file_matches_filter(f["fileName"], name_pat, also_subs)
            ]
            print(
                f"After name filter: {len(planned)} of {before} category-matched files.",
                flush=True,
            )
        else:
            print(f"Files matching categories: {len(planned)}", flush=True)

        for rec in planned:
            name = rec["fileName"]
            cat = category_of(rec)
            url = pick_download_url(rec)
            if not url:
                print(f"SKIP (no URL): {name}", flush=True)
                continue
            rel = subdirs.get(cat, cat.lower())
            out_path = os.path.join(dest_root, pxd, rel, name)
            if os.path.isfile(out_path):
                print(f"EXISTS skip: {out_path}", flush=True)
                continue
            size_b = rec.get("fileSizeBytes") or 0
            print(
                f"GET [{cat}] {name} ({size_b / 1e9:.2f} GB nominal)\n  -> {out_path}",
                flush=True,
            )
            if args.dry_run:
                continue
            rc = run_wget(url, out_path, args.wget)
            if rc != 0:
                print(f"wget failed rc={rc} for {name}", file=sys.stderr)
                return rc
            done += 1
            if max_files and done >= max_files:
                print(f"--max-files {max_files} reached; stopping.", flush=True)
                return 0

    print("\nAll requested downloads finished.", flush=True)

    species = args.species or ("human" if args.human_only else None)
    if species is not None and not args.dry_run:
        print(f"\n=== Extracting per-species MaxQuant tables for {species} ===",
              flush=True)
        # Search.zip lives under PXD014877/search_results — that's also where
        # we want the per-species subdir to land.
        sr_root = os.path.join(dest_root, "PXD014877", "search_results")
        zip_path = os.path.join(sr_root, "Search.zip")
        extract_species_msms(zip_path, species, sr_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
