import requests
import tqdm
import os
import zipfile
import tarfile

from pathlib import Path
from urllib.parse import urlparse


def download_file(url: str, save_dir: str, filename: str = None, extract: bool = False, extract_to: str = None, redownload: bool = False) -> str:
    """
    Download a file from a URL and save it to a specified directory.

    Args:
        url (str): The URL of the file to download.
        save_dir (str): The directory where the file should be saved.
        filename (str, optional): The name to save the file as. If None, the original filename from the URL will be used.
        extract (bool, optional): Whether to extract the downloaded file if it's compressed.
        extract_to (str, optional): The directory where the contents should be extracted. If None, it will be extracted to a directory with the same name as the file (without extension).
        redownload (bool, optional): Whether to redownload the file if it already exists.
    Returns:
        str: The path to the downloaded file.
    """
    # Ensure the save directory exists, if not, create it
    tgt_path = Path(save_dir)
    tgt_path.mkdir(parents=True, exist_ok=True)

    save_path = tgt_path / (
        filename if filename else os.path.basename(urlparse(url).path)
    )

    if not redownload and save_path.exists():
        print(f"File already exists: {save_path}")
        if extract:
            return str(extract_compressed_file(save_path, extract_to=extract_to))
        return str(save_path)

    try:
        # Stream the download to handle large files
        with requests.get(url, stream=True) as r:
            r.raise_for_status()  # Check if the request was successful
            total_size = int(r.headers.get("content-length", 0))
            chunk_size = 8192  # 8 KB
            bytes_downloaded = 0

            with open(save_path, "wb") as f:
                for chunk in tqdm.tqdm(
                    r.iter_content(chunk_size=chunk_size),
                    total=total_size // chunk_size,
                    unit="KB",
                    desc=f"Downloading {filename or os.path.basename(urlparse(url).path)}",):
                    f.write(chunk)
                    bytes_downloaded += len(chunk)
        
        print(20 * "-")
        print(f"Download completed: {save_path} ({bytes_downloaded / (1024 * 1024):.2f} MB)")
        if extract:
            return str(extract_compressed_file(save_path, extract_to=extract_to))
        return str(save_path)
    
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error occurred: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"An error occurred: {req_err}")
    
    return None

def extract_compressed_file(file_path: str, extract_to: str = None) -> str:
    """
    Extract a compressed file (e.g., zip, tar.gz) to a specified directory.

    Args:
        file_path (str): The path to the compressed file.
        extract_to (str): The directory where the contents should be extracted.

    Returns:
        str: The path to the extracted directory.
    """
    
    # Ensure the extract directory exists, if not, create it
    # If the extract path is not provided create a directory with the same name as the file (without extension)
    if not extract_to:
        extract_to = Path(file_path).with_suffix("")
    else:
        extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    try:
        if zipfile.is_zipfile(file_path):
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                members = zip_ref.infolist()
                for member in tqdm.tqdm(
                    members,
                    total=len(members),
                    desc=f"Extracting {Path(file_path).name}",
                    unit="file",
                ):
                    zip_ref.extract(member, path=extract_to)

            print(f"Extracted {file_path} to {extract_to}")
            return str(extract_to)

        elif tarfile.is_tarfile(file_path):
            with tarfile.open(file_path, "r:*") as tar_ref:
                members = tar_ref.getmembers()
                for member in tqdm.tqdm(
                    members,
                    total=len(members),
                    desc=f"Extracting {Path(file_path).name}",
                    unit="file",
                ):
                    tar_ref.extract(member, path=extract_to)

            print(f"Extracted {file_path} to {extract_to}")
            return str(extract_to)

        else:
            print(f"Unsupported file format for: {file_path}")
            return None

    except Exception as e:
        print(f"An error occurred while extracting {file_path}: {e}")
        return None