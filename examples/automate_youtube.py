"""Py-Nerve YouTube Automation Demo.

This script demonstrates how to use Py-Nerve to automate opening a browser,
navigating to YouTube, searching for 'Dandelions', and playing the song.

Usage:
    python examples/automate_youtube.py
"""

from __future__ import annotations

import logging
import time
import webbrowser

import pynerve as nv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pynerve.youtube")


def main() -> None:
    logger.info("Starting Py-Nerve YouTube Automation Demo...")

    # 1. Configure Py-Nerve
    nv.configure(confidence=70, move_duration=0.4)

    # 2. Open the default browser to YouTube
    url = "https://www.youtube.com"
    logger.info(f"Opening browser to {url}...")
    webbrowser.open(url)

    # Give browser window a moment to initialize and bring it to foreground
    time.sleep(2.0)
    nv.focus_window("YouTube")

    # 3. Wait for YouTube to load dynamically using native library wait_for API
    logger.info("Waiting for YouTube page and search field to load...")
    try:
        nv.wait_for("Search", timeout=30.0)
        logger.info("YouTube loaded successfully!")

        # 4. Find the YouTube search box and type the song name
        logger.info("Typing search query into the search input...")
        nv.type_into("Search", "Ruth B - Dandelions", clear=True)
        logger.info("Typed search query. Submitting search...")

        # 5. Press Enter to submit the search
        nv.press_key("enter")

        # Wait for search results to load dynamically
        logger.info("Waiting for search results to appear...")
        nv.wait_for("Dandelions", timeout=15.0)
        logger.info("Search results loaded!")

        # 6. Click on the first search result containing "Dandelions"
        logger.info("Clicking the video...")
        nv.click("Dandelions")
        logger.info("Successfully clicked the video! Playing song 'Dandelions'...")

    except nv.ElementNotFoundError as e:
        logger.error(f"Automation failed: Element not found: {e}")
        logger.info("Make sure the browser window is visible and active on the primary screen.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
