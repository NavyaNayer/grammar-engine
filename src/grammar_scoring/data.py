"""Dataset discovery and loading.

The SHL competition ships a directory that looks like::

    <root>/
        train.csv                 filename,label
        test.csv                  filename,label   (labels are random placeholders)
        sample_submission.csv     filename,label
        audios/
            train/*.wav
            test/*.wav

Kaggle downloads sometimes nest that tree one or two levels deeper (``Dataset/``,
``audios_train/`` ...), so the loader searches for the CSVs and resolves each audio
file against a set of candidate directories rather than assuming a fixed layout.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

AUDIO_SUFFIXES = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

#: Column aliases seen across mirrors of the dataset.
FILENAME_ALIASES = ("filename", "file_name", "file", "audio", "audio_file", "id", "path")
LABEL_ALIASES = ("label", "grammar", "score", "grammar_score", "target", "mos", "y")


class DatasetNotFoundError(FileNotFoundError):
    """Raised when neither the real nor a generated dataset can be located."""


@dataclass(frozen=True)
class DatasetSplit:
    """One split of the competition data."""

    name: str
    frame: pd.DataFrame  # columns: filename, audio_path, label (label may be NaN)

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def filenames(self) -> list[str]:
        return self.frame["filename"].tolist()

    @property
    def audio_paths(self) -> list[Path]:
        return [Path(p) for p in self.frame["audio_path"]]

    @property
    def labels(self):
        return self.frame["label"].to_numpy()


@dataclass(frozen=True)
class Dataset:
    """Train + test splits plus the submission template."""

    root: Path
    train: DatasetSplit
    test: DatasetSplit
    sample_submission: pd.DataFrame | None = None
    #: True when the data was produced by :mod:`grammar_scoring.synthetic`.
    is_synthetic: bool = False

    def describe(self) -> str:
        kind = "synthetic (generated locally)" if self.is_synthetic else "real"
        return (
            f"{kind} dataset at {self.root}\n"
            f"  train: {len(self.train)} samples, "
            f"labels {self.train.labels.min():.2f}-{self.train.labels.max():.2f}\n"
            f"  test : {len(self.test)} samples"
        )


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename whichever columns hold the filename and the label to a canonical name."""
    lowered = {c.lower().strip(): c for c in frame.columns}

    filename_col = next((lowered[a] for a in FILENAME_ALIASES if a in lowered), None)
    if filename_col is None:
        # Fall back to the first object-typed column.
        object_cols = [c for c in frame.columns if frame[c].dtype == object]
        if not object_cols:
            raise ValueError(f"Could not identify a filename column in {list(frame.columns)}")
        filename_col = object_cols[0]

    label_col = next(
        (lowered[a] for a in LABEL_ALIASES if a in lowered and lowered[a] != filename_col), None
    )
    if label_col is None:
        numeric_cols = [
            c
            for c in frame.columns
            if c != filename_col and pd.api.types.is_numeric_dtype(frame[c])
        ]
        label_col = numeric_cols[0] if numeric_cols else None

    renamed = frame.rename(columns={filename_col: "filename"})
    if label_col is not None:
        renamed = renamed.rename(columns={label_col: "label"})
    else:
        renamed["label"] = float("nan")

    renamed["filename"] = renamed["filename"].astype(str).str.strip()
    renamed["label"] = pd.to_numeric(renamed["label"], errors="coerce")
    return renamed[["filename", "label"]]


def _find_csv(root: Path, stem: str) -> Path | None:
    """Locate ``<stem>.csv`` anywhere beneath ``root``, preferring shallow matches."""
    matches = sorted(root.rglob(f"{stem}.csv"), key=lambda p: len(p.relative_to(root).parts))
    if matches:
        return matches[0]
    # Some mirrors use e.g. ``Train.csv`` or ``train_data.csv``.
    loose = [
        p
        for p in sorted(root.rglob("*.csv"), key=lambda p: len(p.relative_to(root).parts))
        if p.stem.lower().replace("_data", "").replace("-", "_") == stem
    ]
    return loose[0] if loose else None


@lru_cache(maxsize=8)
def _audio_index(root: Path) -> dict[str, list[Path]]:
    """Map every audio basename under ``root`` to the paths that carry it."""
    index: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if path.suffix.lower() in AUDIO_SUFFIXES and path.is_file():
            index.setdefault(path.name.lower(), []).append(path)
            index.setdefault(path.stem.lower(), []).append(path)
    return index


def _resolve_audio(root: Path, split: str, filename: str) -> Path | None:
    """Resolve a CSV filename entry to a real file on disk."""
    candidate = Path(filename)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    direct = root / filename
    if direct.exists():
        return direct

    for sub in (f"audios/{split}", f"audio/{split}", split, f"audios_{split}", "audios", "audio"):
        probe = root / sub / candidate.name
        if probe.exists():
            return probe
        if not candidate.suffix:
            for suffix in AUDIO_SUFFIXES:
                probe = root / sub / f"{candidate.name}{suffix}"
                if probe.exists():
                    return probe

    index = _audio_index(root)
    for key in (candidate.name.lower(), candidate.stem.lower()):
        hits = index.get(key)
        if not hits:
            continue
        # Prefer a hit whose path mentions the split (train/test) to avoid collisions.
        split_hits = [h for h in hits if split in {part.lower() for part in h.parts}]
        return (split_hits or hits)[0]
    return None


def _load_split(root: Path, name: str, require_labels: bool) -> DatasetSplit:
    csv_path = _find_csv(root, name)
    if csv_path is None:
        raise DatasetNotFoundError(f"No {name}.csv found under {root}")

    frame = _normalise_columns(pd.read_csv(csv_path))
    resolved, missing = [], []
    for filename in frame["filename"]:
        path = _resolve_audio(root, name, filename)
        if path is None:
            missing.append(filename)
            resolved.append(None)
        else:
            resolved.append(str(path))
    frame["audio_path"] = resolved

    if missing:
        LOGGER.warning(
            "%d/%d %s audio files could not be resolved (e.g. %s); they are dropped.",
            len(missing),
            len(frame),
            name,
            missing[:3],
        )
        frame = frame[frame["audio_path"].notna()].reset_index(drop=True)

    if require_labels:
        before = len(frame)
        frame = frame[frame["label"].notna()].reset_index(drop=True)
        if len(frame) < before:
            LOGGER.warning("Dropped %d %s rows with a missing label.", before - len(frame), name)

    if frame.empty:
        raise DatasetNotFoundError(f"{name} split resolved to zero usable rows under {root}")

    return DatasetSplit(name=name, frame=frame[["filename", "audio_path", "label"]])


def load_dataset(root: str | Path, is_synthetic: bool = False) -> Dataset:
    """Load the competition data rooted at ``root``."""
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise DatasetNotFoundError(f"Dataset root does not exist: {root}")

    train = _load_split(root, "train", require_labels=True)
    test = _load_split(root, "test", require_labels=False)

    submission_path = _find_csv(root, "sample_submission")
    submission = pd.read_csv(submission_path) if submission_path else None

    return Dataset(
        root=root,
        train=train,
        test=test,
        sample_submission=submission,
        is_synthetic=is_synthetic or "synthetic" in root.parts,
    )


def resolve_dataset(
    data_dir: str | Path = "data/raw",
    synthetic_dir: str | Path = "data/synthetic",
    allow_synthetic_fallback: bool = True,
    **synthetic_kwargs,
) -> Dataset:
    """Load the real dataset, generating a synthetic stand-in when it is absent.

    The real SHL data requires Kaggle competition access. When it is not present the
    pipeline falls back to a locally generated speech corpus so that every stage --
    including ASR -- still executes on genuine waveforms.
    """
    data_dir = Path(data_dir)
    try:
        dataset = load_dataset(data_dir)
        LOGGER.info("Loaded real dataset from %s", data_dir)
        return dataset
    except (DatasetNotFoundError, ValueError) as exc:
        if not allow_synthetic_fallback:
            raise
        LOGGER.warning("Real dataset unavailable (%s); falling back to synthetic corpus.", exc)

    from grammar_scoring.synthetic import build_synthetic_dataset

    return build_synthetic_dataset(output_dir=synthetic_dir, **synthetic_kwargs)
