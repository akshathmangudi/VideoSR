import os
import requests
import zipfile


def download_file(url, path, logger):
    """Download a file from a URL to a specified path in chunks."""
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()  # Check for download errors
            with open(path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
        logger.info(f"File downloaded to {path}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading file: {e}")
        raise


def extract_zip(zip_path, extract_dir, logger):
    """Extract a ZIP file to a specified directory."""
    try:
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        logger.info(f"Files extracted to {extract_dir}")
    except zipfile.BadZipFile as e:
        logger.error(f"Error extracting file: {e}")
        raise
