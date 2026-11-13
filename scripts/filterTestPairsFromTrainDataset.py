#!/usr/bin/env python3
"""Remove test-set image/JSON pairs from a train dataset.

Why
---
Deliveries are frequently versioned by copying frames between a *train* and a
*test* dataset.  Frames left in both splits leak test data into training.  This
tool finds every image/JSON pair in the test set that also exists in the train
set and removes the train-side pair, then writes:

* a CSV with one row per **test** image (``--csv-name``; the run timestamp is
  appended to the file name and the image/JSON columns hold base names only,
  with the series carried by its own column), and
* a detailed log with the inputs, progress, per-series summary and totals
  (``--log-name``).

Matching
--------
Identity is decided by ``--match-mode``:

``name``
    Normalised relative name.  The series prefix (``hhus-basilicVein-``,
    ``UGVA2607-``) and the frame prefix (``frm_``, ``frame_``) are stripped,
    then ``<series>/<frame>`` is compared.  Catches re-encoded copies whose
    bytes differ (common when a delivery is re-JPEGed).
``content``
    MD5 of the image bytes.  Catches copies renamed into another series.
``both`` (default)
    Union of the two.

Name-only matches are confirmed pixel-by-pixel by default
(``--verify-pixel``) so a coincidental frame-number collision in the same
series is not deleted.  Content matches are byte-identical and skip the check.

Deletion
--------
By default the train-side image and JSON are unlinked.  Use ``--dry-run`` to
only report, or ``--trash-dir`` to move them into a mirror tree instead of
deleting.  Removals inside a symlinked train series are logged as warnings
because they affect the link target.

Usage
-----
::

    python filterTestPairsFromTrainDataset.py \
        --test-dir  /mnt/.../Testset_hhus_basilicVein_v1.0.261102 \
        --train-dir /mnt/.../basilicVein_v03.260924 \
        --outdir    ./filter-reports

Preview without touching the train set::

    python filterTestPairsFromTrainDataset.py \
        --test-dir ... --train-dir ... --dry-run

Quick single-series debug run::

    python filterTestPairsFromTrainDataset.py \
        --test-dir ... --train-dir ... --limit 20 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_NAME = "filterTestPairs"

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif", ".webp"}
)

# Reserved CSV column for the test-side image path.
IMAGE_COLUMN = "image-name"

CSV_COLUMNS: Tuple[str, ...] = (
    IMAGE_COLUMN,
    "series",
    "json-name",
    "found_in_train",
    "match_type",
    "train_image_matches",
    "train_json_matches",
    "images_removed",
    "jsons_removed",
    "status",
    "note",
)

# Series / frame naming prefixes stripped before comparison.  Override with
# --series-prefix / --frame-prefix when working with another delivery.
DEFAULT_SERIES_PREFIXES: Tuple[str, ...] = ("hhus-basilicVein-", "UGVA2607-")
DEFAULT_FRAME_PREFIXES: Tuple[str, ...] = ("frm_", "frame_")

DEFAULT_CSV_NAME = "test_pair_filter_report.csv"
DEFAULT_PIXEL_TOLERANCE = 12
DEFAULT_PROGRESS_EVERY = 100

MATCH_MODES = ("name", "content", "both")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class _ConsoleFilter(logging.Filter):
    """Allow only START / INPUT / PROGRESS / RESULT / END records on the console."""

    def filter(self, record: logging.LogRecord) -> bool:
        return bool(getattr(record, "console", False))


def setup_logging(log_path: Path) -> logging.Logger:
    """Detailed file logging (DEBUG) + concise five-stage console output."""
    logger = logging.getLogger(APP_NAME)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
    )
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    console_handler.addFilter(_ConsoleFilter())
    logger.addHandler(console_handler)

    return logger


def console_log(
    logger: logging.Logger, message: str, level: int = logging.INFO
) -> None:
    """Emit a message to both the log file and the console."""
    logger.log(level, message, extra={"console": True})


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------


def file_md5(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the MD5 hex digest of a file."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamped_name(name: str, timestamp: str) -> str:
    """Append a run timestamp to a file name, keeping its extension."""
    path = Path(name)
    return f"{path.stem}-{timestamp}{path.suffix or '.csv'}"


def _strip_prefix(value: str, prefixes: Sequence[str]) -> str:
    """Strip the first matching prefix, if any."""
    for prefix in prefixes:
        if value.startswith(prefix):
            return value[len(prefix) :]
    return value


def normalize_key(
    rel_image: str,
    series_prefixes: Sequence[str],
    frame_prefixes: Sequence[str],
) -> str:
    """Build the ``<series>/<frame>`` identity key for a relative image path."""
    parts = Path(rel_image).parts
    series = ""
    if len(parts) > 1:
        series = _strip_prefix(parts[0], series_prefixes)
    stem = _strip_prefix(Path(parts[-1]).stem, frame_prefixes)
    return f"{series}/{stem}" if series else stem


def pixel_identical(a: Path, b: Path, tolerance: int) -> Tuple[bool, str]:
    """Check whether two images are near-identical (guards name collisions)."""
    try:
        import cv2
    except ImportError:  # pragma: no cover - cv2 is a hard dependency in practice
        return True, "opencv not installed; name match accepted unverified"

    image_a = cv2.imread(str(a))
    image_b = cv2.imread(str(b))
    if image_a is None or image_b is None:
        return False, "image unreadable"
    if image_a.shape != image_b.shape:
        return False, f"shape differs {image_a.shape} != {image_b.shape}"

    diff = cv2.absdiff(image_a, image_b)
    worst = int(diff.max())
    return worst <= tolerance, f"max pixel diff={worst} (tolerance={tolerance})"


# ---------------------------------------------------------------------------
# Pair discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pair:
    """An image and its same-stem JSON annotation (JSON may be absent)."""

    image: Path
    json: Optional[Path]
    rel_image: str
    rel_json: Optional[str]
    series: str


class PairCollector:
    """Walk a dataset root and collect image/JSON pairs."""

    def __init__(
        self,
        root: Path,
        follow_symlinks: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.follow_symlinks = follow_symlinks
        self.logger = logger or logging.getLogger(APP_NAME)
        self.symlink_series: Set[str] = set()

    def collect(self) -> List[Pair]:
        """Return all pairs, sorted by relative image path."""
        pairs: List[Pair] = []

        for dirpath, _dirnames, filenames in os.walk(
            self.root, followlinks=self.follow_symlinks
        ):
            directory = Path(dirpath)
            if directory.is_symlink():
                self.symlink_series.add(self._series_name(directory))

            images: Dict[str, Path] = {}
            jsons: Dict[str, Path] = {}
            for filename in filenames:
                path = directory / filename
                suffix = path.suffix.lower()
                if suffix in IMAGE_EXTENSIONS:
                    images[path.stem] = path
                elif suffix == ".json":
                    jsons[path.stem] = path

            for stem in sorted(images):
                image = images[stem]
                json_path = jsons.get(stem)
                rel_image = str(image.relative_to(self.root))
                rel_json = (
                    str(json_path.relative_to(self.root)) if json_path else None
                )
                pairs.append(
                    Pair(
                        image=image,
                        json=json_path,
                        rel_image=rel_image,
                        rel_json=rel_json,
                        series=self._series_name(directory),
                    )
                )

        pairs.sort(key=lambda pair: pair.rel_image)
        return pairs

    def _series_name(self, directory: Path) -> str:
        try:
            relative = directory.relative_to(self.root)
        except ValueError:
            return ""
        return relative.parts[0] if relative.parts else ""


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


@dataclass
class TrainIndex:
    """Lookup tables over the train pairs."""

    by_name: Dict[str, List[Pair]] = field(default_factory=dict)
    by_md5: Dict[str, List[Pair]] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        pairs: Sequence[Pair],
        series_prefixes: Sequence[str],
        frame_prefixes: Sequence[str],
        hash_images: bool,
        logger: logging.Logger,
        progress_every: int = 500,
    ) -> "TrainIndex":
        index = cls()

        for pair in pairs:
            key = normalize_key(pair.rel_image, series_prefixes, frame_prefixes)
            index.by_name.setdefault(key, []).append(pair)

        if hash_images:
            total = len(pairs)
            for position, pair in enumerate(pairs, start=1):
                try:
                    index.by_md5.setdefault(file_md5(pair.image), []).append(pair)
                except OSError as exc:
                    logger.error("cannot hash %s: %s", pair.image, exc)
                if position % progress_every == 0 or position == total:
                    logger.info("HASH | %d/%d train images hashed", position, total)

        return index


@dataclass
class MatchResult:
    """Outcome of matching one test pair against the train index."""

    found: bool
    match_types: Tuple[str, ...]
    train_pairs: List[Pair]
    notes: List[str] = field(default_factory=list)


class PairMatcher:
    """Match test pairs against a pre-built train index."""

    def __init__(
        self,
        index: TrainIndex,
        mode: str,
        series_prefixes: Sequence[str],
        frame_prefixes: Sequence[str],
        verify_pixel: bool,
        pixel_tolerance: int,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.index = index
        self.mode = mode
        self.series_prefixes = series_prefixes
        self.frame_prefixes = frame_prefixes
        self.verify_pixel = verify_pixel
        self.pixel_tolerance = pixel_tolerance
        self.logger = logger or logging.getLogger(APP_NAME)

    def match(self, pair: Pair) -> MatchResult:
        content_hits: List[Pair] = []
        name_hits: List[Pair] = []

        if self.mode in ("content", "both"):
            try:
                content_hits = self.index.by_md5.get(file_md5(pair.image), [])
            except OSError as exc:
                self.logger.error("cannot hash test image %s: %s", pair.image, exc)

        if self.mode in ("name", "both"):
            key = normalize_key(
                pair.rel_image, self.series_prefixes, self.frame_prefixes
            )
            name_hits = self.index.by_name.get(key, [])

        accepted: Dict[str, Pair] = {}
        match_types: Set[str] = set()
        notes: List[str] = []

        for hit in content_hits:
            accepted[hit.rel_image] = hit
            match_types.add("content")

        rejected = 0
        for hit in name_hits:
            if hit.rel_image in accepted:
                continue
            if self.verify_pixel:
                identical, note = pixel_identical(
                    pair.image, hit.image, self.pixel_tolerance
                )
                if not identical:
                    rejected += 1
                    notes.append(
                        f"rejected name match {hit.rel_image}: {note}"
                    )
                    continue
            accepted[hit.rel_image] = hit
            match_types.add("name")

        if not accepted and rejected:
            match_types.add("name_rejected")

        ordered = sorted(accepted.values(), key=lambda item: item.rel_image)
        return MatchResult(
            found=bool(ordered),
            match_types=tuple(sorted(match_types)),
            train_pairs=ordered,
            notes=notes,
        )


# ---------------------------------------------------------------------------
# Removal
# ---------------------------------------------------------------------------


class PairRemover:
    """Delete (or move) train-side files, de-duplicating across test rows."""

    def __init__(
        self,
        dry_run: bool,
        trash_dir: Optional[Path],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.dry_run = dry_run
        self.trash_dir = Path(trash_dir).expanduser().resolve() if trash_dir else None
        self.logger = logger or logging.getLogger(APP_NAME)
        self._done: Set[str] = set()

    def remove_image(self, pair: Pair) -> Tuple[bool, str]:
        return self._remove(pair.image, pair.rel_image)

    def remove_json(self, pair: Pair) -> Tuple[bool, str]:
        if pair.json is None or pair.rel_json is None:
            return False, "train JSON missing"
        return self._remove(pair.json, pair.rel_json)

    def _remove(self, path: Path, rel_path: str) -> Tuple[bool, str]:
        if rel_path in self._done:
            return False, "already removed by an earlier test row"
        if not path.exists():
            return False, "file missing"

        if self.dry_run:
            self._done.add(rel_path)
            return True, "would remove"

        try:
            if self.trash_dir is not None:
                destination = self.trash_dir / rel_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(destination))
                detail = f"moved to {destination}"
            else:
                path.unlink()
                detail = ""
        except OSError as exc:
            self.logger.error("failed to remove %s: %s", path, exc)
            return False, f"remove failed: {exc}"

        self._done.add(rel_path)
        return True, detail


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


class CsvTableWriter:
    """Write the per-test-image report table."""

    def __init__(
        self,
        path: Path,
        columns: Sequence[str],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.path = path
        self.columns = tuple(columns)
        self.logger = logger or logging.getLogger(APP_NAME)

    def write(self, rows: Sequence[Dict[str, str]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.columns))
            writer.writeheader()
            writer.writerows(rows)

        self.logger.info("CSV | wrote %d rows to %s", len(rows), self.path)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


class FilterApp:
    """Orchestrates discovery, matching, removal and reporting."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    def run(self) -> int:
        test_dir = Path(self.args.test_dir).expanduser().resolve()
        train_dir = Path(self.args.train_dir).expanduser().resolve()

        if not test_dir.is_dir():
            print(f"ERROR: test folder not found: {test_dir}", file=sys.stderr)
            return 2
        if not train_dir.is_dir():
            print(f"ERROR: train folder not found: {train_dir}", file=sys.stderr)
            return 2

        out_dir = (
            Path(self.args.outdir).expanduser().resolve()
            if self.args.outdir
            else Path.cwd()
        )
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            print(
                f"ERROR: cannot create output folder {out_dir}: {exc}",
                file=sys.stderr,
            )
            return 2

        timestamp = datetime.now().strftime("%y%m%dT%H%M%S")
        csv_path = out_dir / _timestamped_name(self.args.csv_name, timestamp)
        log_path = out_dir / (
            self.args.log_name or f"{APP_NAME}-{timestamp}.log"
        )

        logger = setup_logging(log_path)
        started = time.perf_counter()

        test_pairs = PairCollector(
            test_dir,
            follow_symlinks=not self.args.no_symlinks,
            logger=logger,
        ).collect()
        train_collector = PairCollector(
            train_dir,
            follow_symlinks=not self.args.no_symlinks,
            logger=logger,
        )
        train_pairs = train_collector.collect()

        console_log(
            logger,
            "START | app={} | test_dir={} | train_dir={} | mode={} | "
            "verify_pixel={} | dry_run={} | trash_dir={} | limit={} | "
            "csv={} | log={}".format(
                APP_NAME,
                test_dir,
                train_dir,
                self.args.match_mode,
                self.args.verify_pixel,
                self.args.dry_run,
                self.args.trash_dir or "<unlink>",
                self.args.limit or "<all>",
                csv_path,
                log_path,
            ),
        )
        console_log(
            logger,
            "INPUT | test_pairs={} | test_series={} | train_pairs={} | "
            "train_series={} | train_symlink_series={}".format(
                len(test_pairs),
                len({pair.series for pair in test_pairs}),
                len(train_pairs),
                len({pair.series for pair in train_pairs}),
                len(train_collector.symlink_series),
            ),
        )
        for series in sorted(train_collector.symlink_series):
            logger.warning(
                "train series is a symlink; removals affect its target: %s", series
            )

        if not test_pairs:
            console_log(logger, "END | status=NO_INPUT | elapsed=0.00s")
            return 1

        index = TrainIndex.build(
            pairs=train_pairs,
            series_prefixes=tuple(self.args.series_prefix),
            frame_prefixes=tuple(self.args.frame_prefix),
            hash_images=self.args.match_mode in ("content", "both"),
            logger=logger,
            progress_every=500,
        )
        matcher = PairMatcher(
            index=index,
            mode=self.args.match_mode,
            series_prefixes=tuple(self.args.series_prefix),
            frame_prefixes=tuple(self.args.frame_prefix),
            verify_pixel=self.args.verify_pixel,
            pixel_tolerance=self.args.pixel_tolerance,
            logger=logger,
        )
        remover = PairRemover(
            dry_run=self.args.dry_run,
            trash_dir=Path(self.args.trash_dir) if self.args.trash_dir else None,
            logger=logger,
        )

        selected = (
            test_pairs[: self.args.limit] if self.args.limit else test_pairs
        )
        total = len(selected)

        rows: List[Dict[str, str]] = []
        counters: Counter = Counter()
        per_series: Dict[str, Counter] = defaultdict(Counter)

        for position, pair in enumerate(selected, start=1):
            result = matcher.match(pair)
            row = self._build_row(pair, result, remover, counters)
            rows.append(row)

            series_counter = per_series[pair.series]
            series_counter["test"] += 1
            series_counter[row["status"]] += 1
            series_counter["removed_images"] += int(row["images_removed"])
            series_counter["removed_jsons"] += int(row["jsons_removed"])

            if position % self.args.progress_every == 0 or position == total:
                console_log(
                    logger,
                    "PROGRESS | {}/{} ({:5.1f}%) | found={} | removed_images={} "
                    "| removed_jsons={} | not_found={} | name_rejected={} "
                    "| errors={}".format(
                        position,
                        total,
                        100.0 * position / total,
                        counters["found"],
                        counters["removed_images"],
                        counters["removed_jsons"],
                        counters["not_found"],
                        counters["name_rejected"],
                        counters["errors"],
                    ),
                )

        CsvTableWriter(csv_path, CSV_COLUMNS, logger).write(rows)

        console_log(
            logger,
            "RESULT | test_pairs={} | found_in_train={} | not_found={} | "
            "name_rejected={} | errors={}".format(
                total,
                counters["found"],
                counters["not_found"],
                counters["name_rejected"],
                counters["errors"],
            ),
        )
        console_log(
            logger,
            "RESULT | removed | images={} | jsons={} | already_removed={} | "
            "json_missing={} | dry_run={}".format(
                counters["removed_images"],
                counters["removed_jsons"],
                counters["already_removed"],
                counters["json_missing"],
                self.args.dry_run,
            ),
        )
        console_log(
            logger, "RESULT | csv={} | rows={}".format(csv_path, len(rows))
        )

        for series in sorted(per_series):
            series_counter = per_series[series]
            console_log(
                logger,
                "RESULT | series | {} | test={} | found={} | removed={} | "
                "not_found={}".format(
                    series or "<root>",
                    series_counter["test"],
                    series_counter["REMOVED"]
                    + series_counter["DRY_RUN"]
                    + series_counter["ALREADY_REMOVED"]
                    + series_counter["FOUND_NOT_REMOVED"],
                    series_counter["removed_images"],
                    series_counter["NOT_FOUND"],
                ),
            )

        elapsed = time.perf_counter() - started
        status = (
            "SUCCESS" if counters["errors"] == 0 else "COMPLETED_WITH_ERRORS"
        )
        console_log(
            logger,
            f"END | status={status} | elapsed={elapsed:.2f}s | log={log_path}",
        )
        return 0 if counters["errors"] == 0 else 1

    # -- row construction ----------------------------------------------------

    def _build_row(
        self,
        pair: Pair,
        result: MatchResult,
        remover: PairRemover,
        counters: Counter,
    ) -> Dict[str, str]:
        notes = list(result.notes)
        removed_images = 0
        removed_jsons = 0

        if result.found:
            counters["found"] += 1
            for hit in result.train_pairs:
                image_done, image_note = remover.remove_image(hit)
                if image_done:
                    removed_images += 1
                if image_note:
                    notes.append(f"{hit.rel_image}: {image_note}")

                json_done, json_note = remover.remove_json(hit)
                if json_done:
                    removed_jsons += 1
                if json_note and json_note != "train JSON missing":
                    notes.append(f"{hit.rel_json}: {json_note}")
                elif json_note == "train JSON missing":
                    counters["json_missing"] += 1

        counters["removed_images"] += removed_images
        counters["removed_jsons"] += removed_jsons

        if result.found:
            if self.args.dry_run:
                status = "DRY_RUN"
            elif removed_images:
                status = "REMOVED"
            else:
                status = "ALREADY_REMOVED"
                counters["already_removed"] += 1
        elif "name_rejected" in result.match_types:
            status = "NAME_REJECTED"
            counters["name_rejected"] += 1
        else:
            status = "NOT_FOUND"
            counters["not_found"] += 1

        return {
            IMAGE_COLUMN: pair.image.name,
            "series": pair.series,
            "json-name": pair.json.name if pair.json else "",
            "found_in_train": str(result.found),
            "match_type": "+".join(result.match_types) if result.match_types else "none",
            "train_image_matches": ";".join(
                hit.rel_image for hit in result.train_pairs
            ),
            "train_json_matches": ";".join(
                hit.rel_json or "" for hit in result.train_pairs
            ),
            "images_removed": str(removed_images),
            "jsons_removed": str(removed_jsons),
            "status": status,
            "note": " | ".join(notes),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Remove test-set image/JSON pairs from a train dataset and write a "
            "CSV report plus a detailed log."
        ),
        epilog=(
            "Matching: 'name' strips the series/frame prefixes and compares "
            "<series>/<frame>; 'content' compares image MD5; 'both' unions them."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--test-dir",
        required=True,
        help="Test dataset folder (the split that must stay clean).",
    )
    parser.add_argument(
        "--train-dir",
        required=True,
        help="Train dataset folder from which matching pairs are removed.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Folder for the CSV and log (default: current working directory).",
    )
    parser.add_argument(
        "--csv-name",
        default=DEFAULT_CSV_NAME,
        help=(
            "CSV base file name; a -<timestamp> suffix is appended "
            f"(default: {DEFAULT_CSV_NAME})."
        ),
    )
    parser.add_argument(
        "--log-name",
        default=None,
        help="Log file name (default: filterTestPairs-<timestamp>.log).",
    )
    parser.add_argument(
        "--match-mode",
        choices=MATCH_MODES,
        default="both",
        help="Identity rule to use (default: both).",
    )
    parser.add_argument(
        "--series-prefix",
        action="append",
        default=list(DEFAULT_SERIES_PREFIXES),
        help=(
            "Series-name prefix to strip before comparing; repeatable "
            f"(default: {', '.join(DEFAULT_SERIES_PREFIXES)})."
        ),
    )
    parser.add_argument(
        "--frame-prefix",
        action="append",
        default=list(DEFAULT_FRAME_PREFIXES),
        help=(
            "Frame-name prefix to strip before comparing; repeatable "
            f"(default: {', '.join(DEFAULT_FRAME_PREFIXES)})."
        ),
    )
    parser.add_argument(
        "--verify-pixel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Confirm name-only matches pixel-by-pixel before deleting "
            "(default: enabled)."
        ),
    )
    parser.add_argument(
        "--pixel-tolerance",
        type=int,
        default=DEFAULT_PIXEL_TOLERANCE,
        help=(
            "Maximum per-channel pixel difference accepted as identical for "
            f"name matches (default: {DEFAULT_PIXEL_TOLERANCE})."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report matches without deleting or moving anything.",
    )
    parser.add_argument(
        "--trash-dir",
        default=None,
        help=(
            "Move removed pairs into this mirror tree instead of deleting them."
        ),
    )
    parser.add_argument(
        "--no-symlinks",
        action="store_true",
        help="Do not follow symlinked series folders while walking the datasets.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N test pairs (0 = all).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
        help=f"Emit a PROGRESS line every N test pairs (default: {DEFAULT_PROGRESS_EVERY}).",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.limit < 0:
        parser.error("--limit must be >= 0")
    if args.pixel_tolerance < 0:
        parser.error("--pixel-tolerance must be >= 0")
    if args.progress_every < 1:
        parser.error("--progress-every must be >= 1")

    test_dir = Path(args.test_dir).expanduser().resolve()
    train_dir = Path(args.train_dir).expanduser().resolve()
    if not test_dir.is_dir():
        parser.error(f"--test-dir is not a directory: {test_dir}")
    if not train_dir.is_dir():
        parser.error(f"--train-dir is not a directory: {train_dir}")
    if test_dir == train_dir:
        parser.error("--test-dir and --train-dir must be different folders")
    if train_dir.is_relative_to(test_dir) or test_dir.is_relative_to(train_dir):
        print(
            "WARNING: one dataset folder is inside the other; self-matching is "
            "possible.",
            file=sys.stderr,
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)

    try:
        return FilterApp(args).run()
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        logging.getLogger(APP_NAME).exception("Fatal error: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
