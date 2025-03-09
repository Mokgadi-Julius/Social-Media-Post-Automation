#!/usr/bin/env python3
"""
image_processor.py - Handles image selection and uploading to ImgBB

This script:
1. Selects a random image from a directory
2. Uploads the image to ImgBB and gets a URL
3. Caches URLs to avoid re-uploading the same image
4. Moves used images to a 'used' directory
5. Returns the URL and caption for the image
"""

import os
import sys
import random
import shutil
import json
import logging
import base64
import requests
import time
import hashlib
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="image_processor.log",
)
logger = logging.getLogger(__name__)

# Directory configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "images")
USED_IMAGE_DIR = os.path.join(BASE_DIR, "images", "used")
CAPTIONS_FILE = os.path.join(BASE_DIR, "image_captions.json")
URL_CACHE_FILE = os.path.join(BASE_DIR, "image_url_cache.json")

# Ensure directories exist
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(USED_IMAGE_DIR, exist_ok=True)

# ImgBB API key
IMGBB_API_KEY = os.environ.get("IMGBB_API_KEY", "")


def load_image_captions():
    """Load image captions from JSON file"""
    if os.path.exists(CAPTIONS_FILE):
        try:
            with open(CAPTIONS_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading image captions: {e}")

    # Return empty dict if file doesn't exist or there's an error
    return {}


def save_image_captions(captions):
    """Save image captions to JSON file"""
    try:
        with open(CAPTIONS_FILE, "w") as f:
            json.dump(captions, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving image captions: {e}")
        return False


def load_url_cache():
    """Load the URL cache from file"""
    if os.path.exists(URL_CACHE_FILE):
        try:
            with open(URL_CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading URL cache: {e}")
    return {}


def save_url_cache(cache):
    """Save the URL cache to file"""
    try:
        with open(URL_CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving URL cache: {e}")
        return False


def get_file_hash(file_path):
    """Get a simple hash of a file for caching purposes"""
    if not os.path.exists(file_path):
        return None

    # Get file stats for a simple hash
    file_stat = os.stat(file_path)
    file_info = f"{file_path}_{file_stat.st_size}_{file_stat.st_mtime}"
    return hashlib.md5(file_info.encode()).hexdigest()


def get_image_caption(image_name, captions=None):
    """Get caption for an image, or generate a default one"""
    if captions is None:
        captions = load_image_captions()

    # Return existing caption if available
    if image_name in captions:
        return captions[image_name]

    # Generate a default caption based on filename
    filename_without_ext = os.path.splitext(image_name)[0]
    default_caption = filename_without_ext.replace("_", " ").replace("-", " ").title()

    # Store this caption for future use
    captions[image_name] = default_caption
    save_image_captions(captions)

    return default_caption


def select_random_image():
    """Select a random image from the directory and return its path"""
    try:
        # Get all valid image files
        images = [
            f
            for f in os.listdir(IMAGE_DIR)
            if os.path.isfile(os.path.join(IMAGE_DIR, f))
            and f.lower().endswith((".jpg", ".jpeg", ".png", ".gif"))
            and not os.path.isdir(os.path.join(IMAGE_DIR, f))
        ]

        if not images:
            logger.warning("No images found in the directory.")
            return None

        # Select a random image
        selected_image = random.choice(images)
        image_path = os.path.join(IMAGE_DIR, selected_image)

        logger.info(f"Selected image: {selected_image}")
        return image_path

    except Exception as e:
        logger.error(f"Error selecting random image: {e}")
        return None


def upload_to_imgbb(image_path):
    """Upload an image to ImgBB and return the URL"""
    if not IMGBB_API_KEY:
        logger.error("ImgBB API key not set. Set IMGBB_API_KEY in your .env file.")
        print("ERROR: ImgBB API key not set. Set IMGBB_API_KEY in your .env file.")
        return None

    try:
        logger.info(f"Uploading to ImgBB: {os.path.basename(image_path)}")

        with open(image_path, "rb") as file:
            # Convert to base64
            base64_image = base64.b64encode(file.read()).decode("utf-8")

        url = "https://api.imgbb.com/1/upload"
        payload = {
            "key": IMGBB_API_KEY,
            "image": base64_image,
            "name": os.path.basename(image_path),
        }

        response = requests.post(url, payload)

        if response.status_code == 200:
            json_response = response.json()
            if json_response.get("success", False):
                # Return direct URL to the image
                image_url = json_response["data"]["url"]
                display_url = json_response["data"]["display_url"]
                delete_url = json_response["data"]["delete_url"]
                thumbnail_url = json_response["data"].get("thumb", {}).get("url")

                # Log the URLs
                logger.info(f"ImgBB upload successful: {image_url}")

                # Return the URLs
                return {
                    "url": image_url,
                    "display_url": display_url,
                    "delete_url": delete_url,
                    "thumbnail_url": thumbnail_url,
                }

        # Handle errors
        error_msg = f"ImgBB upload failed: Status code {response.status_code}"
        if response.text:
            try:
                error_data = response.json()
                error_msg += f", Error: {error_data.get('error', {}).get('message', 'Unknown error')}"
            except:
                error_msg += f", Response: {response.text[:100]}"

        logger.error(error_msg)
        print(error_msg)
        return None

    except Exception as e:
        logger.error(f"Error uploading to ImgBB: {e}")
        print(f"Error uploading to ImgBB: {e}")
        return None


def move_used_image(image_path):
    """Move used image to the 'used' directory"""
    try:
        if not image_path or not os.path.exists(image_path):
            logger.warning(f"Image path not found: {image_path}")
            return False

        filename = os.path.basename(image_path)
        new_path = os.path.join(USED_IMAGE_DIR, filename)

        # If image with same name exists in 'used' directory, add timestamp
        if os.path.exists(new_path):
            name, ext = os.path.splitext(filename)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            new_filename = f"{name}_{timestamp}{ext}"
            new_path = os.path.join(USED_IMAGE_DIR, new_filename)

        # Move the file
        shutil.move(image_path, new_path)
        logger.info(f"Moved image to: {new_path}")
        return True

    except Exception as e:
        logger.error(f"Error moving used image: {e}")
        return False


def get_random_image():
    """Main function to get a random image, upload it to ImgBB, and get its caption"""
    # Select a random image
    image_path = select_random_image()

    if not image_path:
        logger.warning("Failed to get random image")
        return None, None, None

    # Get the filename for caption
    image_filename = os.path.basename(image_path)

    # Get the caption
    caption = get_image_caption(image_filename)

    # Check the cache first
    file_hash = get_file_hash(image_path)
    cache = load_url_cache()

    image_url = None

    if file_hash in cache and cache[file_hash].get("url"):
        logger.info(f"Using cached URL for {image_filename}")
        image_url = cache[file_hash]["url"]
    else:
        # Upload to ImgBB
        result = upload_to_imgbb(image_path)

        if result:
            image_url = result["url"]

            # Save to cache
            cache[file_hash] = {
                "url": image_url,
                "display_url": result.get("display_url"),
                "delete_url": result.get("delete_url"),
                "thumbnail_url": result.get("thumbnail_url"),
                "path": image_path,
                "filename": image_filename,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            save_url_cache(cache)

    if not image_url:
        logger.warning(f"Failed to get URL for image: {image_path}")

    return image_path, image_url, caption


if __name__ == "__main__":
    # This allows running the script directly to test
    print("=== Image Processor Test (ImgBB) ===")

    # Make sure API key is set
    if not IMGBB_API_KEY:
        print("ERROR: ImgBB API key not set. Add it to your .env file as IMGBB_API_KEY")
        sys.exit(1)

    image_path, image_url, caption = get_random_image()

    if image_path:
        print(f"Selected image: {os.path.basename(image_path)}")
        print(f"Image caption: {caption}")

        if image_url:
            print(f"Image URL: {image_url}")
            print("\nUpload successful!")

            # Ask if user wants to move the image
            response = input("\nMove image to used directory? (y/n): ")
            if response.lower() == "y":
                if move_used_image(image_path):
                    print("Image moved successfully.")
                else:
                    print("Failed to move image.")
        else:
            print("Failed to upload image to ImgBB.")
    else:
        print("No images available.")
