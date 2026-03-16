"""Tests for the ZeroWaste dataset download module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from zerowaste_bootstrap.data.download import (
    DATASET_FILES,
    DOWNLOAD_CHUNK_SIZE,
    LABELED_DIR,
    LABELED_SPLITS,
    UNLABELED_DIR,
    ZIP_PASSWORD,
    ZENODO_BASE_URL,
    _download_file,
    _extract_zip,
    _get_download_url,
    _is_already_downloaded,
    _verify_labeled_structure,
    _verify_unlabeled_structure,
    download_zerowaste,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_labeled_structure(data_dir: Path) -> None:
    """Populate *data_dir* with the expected labeled dataset structure."""
    labeled_root = data_dir / LABELED_DIR
    for split in LABELED_SPLITS:
        (labeled_root / split / "data").mkdir(parents=True, exist_ok=True)
        (labeled_root / split / "labels.json").write_text("{}")


def _create_unlabeled_structure(data_dir: Path) -> None:
    """Populate *data_dir* with the expected unlabeled dataset directory."""
    (data_dir / UNLABELED_DIR).mkdir(parents=True, exist_ok=True)


def _create_full_structure(data_dir: Path) -> None:
    """Create both labeled and unlabeled structures."""
    _create_labeled_structure(data_dir)
    _create_unlabeled_structure(data_dir)


# ---------------------------------------------------------------------------
# URL helper
# ---------------------------------------------------------------------------

class TestGetDownloadUrl:
    def test_url_format(self):
        url = _get_download_url("zerowaste-f-final.zip")
        assert url == f"{ZENODO_BASE_URL}/zerowaste-f-final.zip/content"

    def test_url_contains_filename(self):
        for fname in DATASET_FILES:
            url = _get_download_url(fname)
            assert fname in url


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

class TestVerifyLabeledStructure:
    def test_valid_structure(self, tmp_path: Path):
        _create_labeled_structure(tmp_path)
        assert _verify_labeled_structure(tmp_path) is True

    def test_missing_data_dir(self, tmp_path: Path):
        labeled_root = tmp_path / LABELED_DIR
        for split in LABELED_SPLITS:
            (labeled_root / split).mkdir(parents=True)
            (labeled_root / split / "labels.json").write_text("{}")
        assert _verify_labeled_structure(tmp_path) is False

    def test_missing_labels_json(self, tmp_path: Path):
        labeled_root = tmp_path / LABELED_DIR
        for split in LABELED_SPLITS:
            (labeled_root / split / "data").mkdir(parents=True)
        assert _verify_labeled_structure(tmp_path) is False

    def test_empty_dir(self, tmp_path: Path):
        assert _verify_labeled_structure(tmp_path) is False

    def test_partial_splits(self, tmp_path: Path):
        """Only train exists, val and test missing."""
        labeled_root = tmp_path / LABELED_DIR
        (labeled_root / "train" / "data").mkdir(parents=True)
        (labeled_root / "train" / "labels.json").write_text("{}")
        assert _verify_labeled_structure(tmp_path) is False


class TestVerifyUnlabeledStructure:
    def test_valid(self, tmp_path: Path):
        _create_unlabeled_structure(tmp_path)
        assert _verify_unlabeled_structure(tmp_path) is True

    def test_missing(self, tmp_path: Path):
        assert _verify_unlabeled_structure(tmp_path) is False


class TestIsAlreadyDownloaded:
    def test_both_present(self, tmp_path: Path):
        _create_full_structure(tmp_path)
        assert _is_already_downloaded(tmp_path) is True

    def test_labeled_only(self, tmp_path: Path):
        _create_labeled_structure(tmp_path)
        assert _is_already_downloaded(tmp_path) is False

    def test_unlabeled_only(self, tmp_path: Path):
        _create_unlabeled_structure(tmp_path)
        assert _is_already_downloaded(tmp_path) is False

    def test_empty(self, tmp_path: Path):
        assert _is_already_downloaded(tmp_path) is False


# ---------------------------------------------------------------------------
# Download file
# ---------------------------------------------------------------------------

class TestDownloadFile:
    def test_streams_content_to_file(self, tmp_path: Path):
        content = b"hello world" * 100
        dest = tmp_path / "test.zip"

        mock_response = MagicMock()
        mock_response.headers = {"content-length": str(len(content))}
        mock_response.iter_content.return_value = [content]
        mock_response.raise_for_status = MagicMock()

        with patch("zerowaste_bootstrap.data.download.requests.get", return_value=mock_response):
            _download_file("https://example.com/file.zip", dest)

        assert dest.read_bytes() == content

    def test_raises_on_http_error(self, tmp_path: Path):
        dest = tmp_path / "test.zip"

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("404 Not Found")

        with patch("zerowaste_bootstrap.data.download.requests.get", return_value=mock_response):
            with pytest.raises(Exception, match="404 Not Found"):
                _download_file("https://example.com/file.zip", dest)

    def test_handles_missing_content_length(self, tmp_path: Path):
        content = b"data"
        dest = tmp_path / "test.zip"

        mock_response = MagicMock()
        mock_response.headers = {}
        mock_response.iter_content.return_value = [content]
        mock_response.raise_for_status = MagicMock()

        with patch("zerowaste_bootstrap.data.download.requests.get", return_value=mock_response):
            _download_file("https://example.com/file.zip", dest)

        assert dest.read_bytes() == content

    def test_multiple_chunks(self, tmp_path: Path):
        chunks = [b"aaa", b"bbb", b"ccc"]
        dest = tmp_path / "test.zip"

        mock_response = MagicMock()
        mock_response.headers = {"content-length": "9"}
        mock_response.iter_content.return_value = chunks
        mock_response.raise_for_status = MagicMock()

        with patch("zerowaste_bootstrap.data.download.requests.get", return_value=mock_response):
            _download_file("https://example.com/file.zip", dest)

        assert dest.read_bytes() == b"aaabbbccc"

    def test_passes_stream_and_timeout(self, tmp_path: Path):
        dest = tmp_path / "test.zip"

        mock_response = MagicMock()
        mock_response.headers = {"content-length": "0"}
        mock_response.iter_content.return_value = []
        mock_response.raise_for_status = MagicMock()

        with patch("zerowaste_bootstrap.data.download.requests.get", return_value=mock_response) as mock_get:
            _download_file("https://example.com/file.zip", dest)

        mock_get.assert_called_once_with(
            "https://example.com/file.zip", stream=True, timeout=60
        )


# ---------------------------------------------------------------------------
# Extract ZIP
# ---------------------------------------------------------------------------

class TestExtractZip:
    def test_calls_unzip_with_password(self, tmp_path: Path):
        zip_path = tmp_path / "test.zip"
        zip_path.touch()
        dest_dir = tmp_path / "output"
        dest_dir.mkdir()

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""

        with patch("zerowaste_bootstrap.data.download.subprocess.run", return_value=mock_result) as mock_run:
            _extract_zip(zip_path, dest_dir)

        mock_run.assert_called_once_with(
            ["unzip", "-o", "-P", ZIP_PASSWORD, str(zip_path), "-d", str(dest_dir)],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_raises_on_failure(self, tmp_path: Path):
        zip_path = tmp_path / "test.zip"
        zip_path.touch()
        dest_dir = tmp_path / "output"
        dest_dir.mkdir()

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "bad password"

        with patch("zerowaste_bootstrap.data.download.subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="unzip failed"):
                _extract_zip(zip_path, dest_dir)


# ---------------------------------------------------------------------------
# Main download_zerowaste function
# ---------------------------------------------------------------------------

class TestDownloadZerowaste:
    def test_idempotent_skips_when_exists(self, tmp_path: Path):
        """When dataset already exists, nothing is downloaded."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _create_full_structure(data_dir)

        with (
            patch("zerowaste_bootstrap.data.download._download_file") as mock_dl,
            patch("zerowaste_bootstrap.data.download._extract_zip") as mock_extract,
        ):
            result = download_zerowaste(data_dir)

        mock_dl.assert_not_called()
        mock_extract.assert_not_called()
        assert result == data_dir.resolve()

    def test_downloads_and_extracts_all_files(self, tmp_path: Path):
        """Full flow: download, extract, verify."""
        data_dir = tmp_path / "data"

        def fake_extract(zip_path: Path, dest_dir: Path) -> None:
            _create_full_structure(dest_dir)

        with (
            patch("zerowaste_bootstrap.data.download._download_file") as mock_dl,
            patch(
                "zerowaste_bootstrap.data.download._extract_zip",
                side_effect=fake_extract,
            ) as mock_extract,
        ):
            result = download_zerowaste(data_dir)

        assert result == data_dir.resolve()
        assert mock_dl.call_count == len(DATASET_FILES)
        assert mock_extract.call_count == len(DATASET_FILES)

        # Verify the correct URLs were used
        for i, fname in enumerate(DATASET_FILES):
            dl_call = mock_dl.call_args_list[i]
            assert fname in dl_call.args[0]

    def test_skips_download_when_zip_cached(self, tmp_path: Path):
        """When ZIP files already exist locally, skip download but still extract."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        for fname in DATASET_FILES:
            (data_dir / fname).write_bytes(b"fake zip content")

        def fake_extract(zip_path: Path, dest_dir: Path) -> None:
            _create_full_structure(dest_dir)

        with (
            patch("zerowaste_bootstrap.data.download._download_file") as mock_dl,
            patch(
                "zerowaste_bootstrap.data.download._extract_zip",
                side_effect=fake_extract,
            ),
        ):
            download_zerowaste(data_dir)

        mock_dl.assert_not_called()

    def test_creates_data_dir_if_missing(self, tmp_path: Path):
        """data_dir is created when it does not exist."""
        data_dir = tmp_path / "nonexistent" / "deep" / "path"

        def fake_extract(zip_path: Path, dest_dir: Path) -> None:
            _create_full_structure(dest_dir)

        with (
            patch("zerowaste_bootstrap.data.download._download_file"),
            patch(
                "zerowaste_bootstrap.data.download._extract_zip",
                side_effect=fake_extract,
            ),
        ):
            result = download_zerowaste(data_dir)

        assert result.exists()
        assert result == data_dir.resolve()

    def test_uses_data_config_default(self, tmp_path: Path):
        """When data_dir is None, falls back to DataConfig."""
        _create_full_structure(tmp_path / "data")

        with patch(
            "zerowaste_bootstrap.data.download.DataConfig",
            return_value=MagicMock(data_dir=tmp_path / "data"),
        ):
            result = download_zerowaste(None)

        assert result == (tmp_path / "data").resolve()

    def test_raises_if_labeled_structure_invalid_after_extract(self, tmp_path: Path):
        """Extraction that does not produce expected structure raises."""
        data_dir = tmp_path / "data"

        def fake_extract_broken(zip_path: Path, dest_dir: Path) -> None:
            # Only create unlabeled, not labeled
            _create_unlabeled_structure(dest_dir)

        with (
            patch("zerowaste_bootstrap.data.download._download_file"),
            patch(
                "zerowaste_bootstrap.data.download._extract_zip",
                side_effect=fake_extract_broken,
            ),
        ):
            with pytest.raises(RuntimeError, match="Labeled dataset structure invalid"):
                download_zerowaste(data_dir)

    def test_raises_if_unlabeled_structure_invalid_after_extract(self, tmp_path: Path):
        """Extraction that does not produce unlabeled dir raises."""
        data_dir = tmp_path / "data"

        def fake_extract_broken(zip_path: Path, dest_dir: Path) -> None:
            # Only create labeled, not unlabeled
            _create_labeled_structure(dest_dir)

        with (
            patch("zerowaste_bootstrap.data.download._download_file"),
            patch(
                "zerowaste_bootstrap.data.download._extract_zip",
                side_effect=fake_extract_broken,
            ),
        ):
            with pytest.raises(RuntimeError, match="Unlabeled dataset structure invalid"):
                download_zerowaste(data_dir)


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------

class TestConstants:
    def test_dataset_files_not_empty(self):
        assert len(DATASET_FILES) >= 2

    def test_zip_password_set(self):
        assert ZIP_PASSWORD

    def test_chunk_size_positive(self):
        assert DOWNLOAD_CHUNK_SIZE > 0

    def test_labeled_splits(self):
        assert set(LABELED_SPLITS) == {"train", "val", "test"}
