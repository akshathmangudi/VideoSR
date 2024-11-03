import os
import config
import requests
import zipfile
import logging
from utils import download_file, extract_zip

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Downloading dataset")
    download_file(config.URL_PATH, config.ZIP_PATH, logger)
    logger.info("Extracting the .zip file...")
    extract_zip(config.ZIP_PATH, config.EXTRACT_PATH, logger)

    try:
        os.remove(config.ZIP_PATH)
        logger.info("Zip file deleted after extraction.")
    except OSError as e:
        logger.error(f"Error deleting zip file: {e}")
