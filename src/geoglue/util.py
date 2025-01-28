"Utility functions for geoglue"

import logging
from pathlib import Path
import shutil

import requests


COMPRESSED_FILE_EXTS = [".tar.gz", ".tar.bz2", ".zip"]


def unpack_file(path: Path, in_folder: Path | None = None):
    """Unpack a zipped file."""
    extract_dir = in_folder or path.parent
    shutil.unpack_archive(path, str(extract_dir))


def download_file(
    url: str, path: Path, unpack: bool = True, unpack_in_folder: Path | None = None
) -> bool:
    """Download a file from a given URL to a given path."""
    if (r := requests.get(url)).status_code == 200:
        with open(path, "wb") as out:
            for bits in r.iter_content():
                out.write(bits)
        # Unpack file
        if unpack and any(str(path).endswith(ext) for ext in COMPRESSED_FILE_EXTS):
            logging.info(f"Unpacking downloaded file {path}")
            unpack_file(path, unpack_in_folder)
        return True
    else:
        logging.error(f"Failed to fetch {url}, status={r.status_code}")
    return False
