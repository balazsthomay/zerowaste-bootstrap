"""Download and extract the ZeroWaste dataset from Zenodo."""

import logging
import subprocess
from pathlib import Path

import requests
from tqdm import tqdm

from zerowaste_bootstrap.config import DataConfig

logger = logging.getLogger(__name__)

ZENODO_BASE_URL = "https://zenodo.org/api/records/6412647/files"
ZIP_PASSWORD = "UP#1VuX409z4"

DATASET_FILES = [
    "zerowaste-f-final.zip",
    "zerowaste-s-final.zip",
]

# Expected directory structure after extraction
LABELED_SPLITS = ("train", "val", "test")
LABELED_DIR = "zerowaste-f"
UNLABELED_DIR = "zerowaste-s"

DOWNLOAD_CHUNK_SIZE = 8192


def _get_download_url(filename: str) -> str:
    """Build the Zenodo download URL for a given filename."""
    return f"{ZENODO_BASE_URL}/{filename}/content"


def _download_file(url: str, dest: Path, desc: str | None = None) -> None:
    """Stream-download a file from *url* to *dest* with a tqdm progress bar.

    Raises ``requests.HTTPError`` on non-2xx responses.
    """
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    label = desc or dest.name

    with (
        open(dest, "wb") as fh,
        tqdm(total=total_size, unit="B", unit_scale=True, desc=label) as pbar,
    ):
        for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
            fh.write(chunk)
            pbar.update(len(chunk))

    logger.info("Downloaded %s (%d bytes)", dest.name, dest.stat().st_size)


def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract a password-protected ZIP into *dest_dir* using ``unzip``."""
    logger.info("Extracting %s -> %s", zip_path.name, dest_dir)
    result = subprocess.run(
        ["unzip", "-o", "-P", ZIP_PASSWORD, str(zip_path), "-d", str(dest_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"unzip failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    logger.info("Extraction complete: %s", zip_path.name)


def _verify_labeled_structure(data_dir: Path) -> bool:
    """Check that the labeled dataset has the expected directory layout."""
    labeled_root = data_dir / LABELED_DIR
    for split in LABELED_SPLITS:
        split_dir = labeled_root / split
        if not (split_dir / "data").is_dir():
            logger.warning("Missing directory: %s", split_dir / "data")
            return False
        if not (split_dir / "labels.json").is_file():
            logger.warning("Missing file: %s", split_dir / "labels.json")
            return False
    return True


def _verify_unlabeled_structure(data_dir: Path) -> bool:
    """Check that the unlabeled dataset directory exists."""
    unlabeled_root = data_dir / UNLABELED_DIR
    if not unlabeled_root.is_dir():
        logger.warning("Missing directory: %s", unlabeled_root)
        return False
    return True


def _is_already_downloaded(data_dir: Path) -> bool:
    """Return True when both labeled and unlabeled data are present."""
    return _verify_labeled_structure(data_dir) and _verify_unlabeled_structure(data_dir)


def download_zerowaste(data_dir: Path | None = None) -> Path:
    """Download and extract the ZeroWaste dataset.

    Parameters
    ----------
    data_dir:
        Root directory where dataset folders will be placed.  When *None*,
        falls back to ``DataConfig().data_dir`` (respects ``ZEROWASTE_DATA_DIR``
        env var).

    Returns
    -------
    Path
        The resolved *data_dir* containing the extracted dataset.
    """
    if data_dir is None:
        data_dir = DataConfig().data_dir

    data_dir = data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    if _is_already_downloaded(data_dir):
        logger.info("Dataset already present at %s — skipping download.", data_dir)
        return data_dir

    logger.info("Downloading ZeroWaste dataset to %s", data_dir)

    for filename in DATASET_FILES:
        zip_dest = data_dir / filename
        url = _get_download_url(filename)

        if not zip_dest.exists():
            logger.info("Downloading %s …", filename)
            _download_file(url, zip_dest, desc=filename)
        else:
            logger.info("ZIP already cached: %s", zip_dest)

        _extract_zip(zip_dest, data_dir)

    if not _verify_labeled_structure(data_dir):
        raise RuntimeError(
            f"Labeled dataset structure invalid after extraction in {data_dir}"
        )
    if not _verify_unlabeled_structure(data_dir):
        raise RuntimeError(
            f"Unlabeled dataset structure invalid after extraction in {data_dir}"
        )

    logger.info("ZeroWaste dataset ready at %s", data_dir)
    return data_dir
