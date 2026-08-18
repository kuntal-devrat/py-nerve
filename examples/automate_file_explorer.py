"""Py-Nerve Windows File Explorer Automation Demo (UIA Backend).

This script demonstrates how to use Py-Nerve to open Windows File Explorer,
focus it, search for a song containing 'SEMPERO', and play it by double-clicking.

Usage:
    python examples/automate_file_explorer.py
"""

from __future__ import annotations

import logging
import os
import subprocess
import time

import pynerve as nv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pynerve.file_explorer")

def main() -> None:
    logger.info("Starting Py-Nerve File Explorer Automation Demo...")

    # 1. Configure Py-Nerve with the accessibility backend
    nv.configure(backend="accessibility", confidence=70, move_duration=0.4)

    # 2. Open File Explorer to the user's Music folder
    music_dir = os.path.normpath(os.path.expanduser("~/Music"))
    logger.info(f"Opening File Explorer to: {music_dir}")
    subprocess.Popen(["explorer.exe", music_dir])

    # Give the File Explorer window a moment to open and initialize
    time.sleep(4.0)

    # 3. Focus the File Explorer window
    # The title of the window opened to ~/Music is usually "Music"
    logger.info("Focusing the File Explorer window...")
    focused = nv.focus_window("Music", timeout=20.0)
    if not focused:
        logger.error("Could not find or focus the 'Music' folder window.")
        return

    # 4. Use search shortcut Ctrl+E to focus the search field and type query
    logger.info("Focusing the search field via Ctrl+E...")
    nv.key_combo(["ctrl", "e"])
    time.sleep(0.5)

    logger.info("Typing search query 'SEMPERO'...")
    # Type the search term and submit with Enter
    nv.type_text("SEMPERO")
    time.sleep(0.2)
    nv.press_key("enter")

    # 5. Wait for the search results to populate
    logger.info("Waiting for search results containing 'SEMPERO' to appear...")
    try:
        # wait_for utilizes our new UIA scoped queries and vision fallback
        logger.info("Locating the target song file item...")
        nv.wait_for("SEMPERO", timeout=10.0)

        # Get all elements containing "SEMPERO" to handle ambiguity (search input vs list items)
        elements = nv.find_all("SEMPERO")
        logger.info(f"Detected matches on screen: {[f'{e.text} at {e.center}' for e in elements]}")

        # Filter: actual file items reside below the top headers (y > 150) and exclude window titles
        file_items = [e for e in elements if e.y > 150 and not e.text.endswith("File Explorer")]
        if not file_items:
            raise nv.ElementNotFoundError("Could not find the actual song file item on screen.")

        target_element = file_items[0]
        logger.info(f"Targeting file item element: '{target_element.text}' at {target_element.center}")

        # 6. Click the song file to select it and press Enter to play it
        logger.info("Clicking the file to select it...")
        nv.click(target_element)
        time.sleep(0.5)
        logger.info("Pressing Enter to play the song...")
        nv.press_key("enter")
        logger.info("Song playback triggered successfully!")

    except nv.ElementNotFoundError as e:
        logger.error(f"Automation failed: Element not found: {e}")
        logger.info("Make sure the File Explorer window is visible and active on the primary screen.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
