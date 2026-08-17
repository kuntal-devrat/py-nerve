"""Basic demo script for Py-Nerve.

This script demonstrates the core API of Py-Nerve.
Run it with a desktop application visible on screen.

Usage:
    python examples/basic_demo.py
"""

from __future__ import annotations

import logging

import pynerve as nv

# Enable debug logging to see what's happening
logging.basicConfig(level=logging.INFO)


def main() -> None:
    print("Py-Nerve Basic Demo")
    print("=" * 40)

    # Configure Py-Nerve
    nv.configure(confidence=75, move_duration=0.3)

    # Get current mouse position
    pos = nv.get_position()
    print(f"Current mouse position: {pos}")

    # Take a screenshot
    print("\nTaking screenshot...")
    img = nv.screenshot()
    print(f"Screenshot size: {img.size}")

    # Find all UI elements on screen
    print("\nScanning for UI elements...")
    # Note: This requires OCR to be initialized (first call may be slow)
    try:
        elements = nv.find_all("File")
        print(f"Found {len(elements)} elements matching 'File'")
        for el in elements:
            print(f"  - '{el.text}' at ({el.x:.0f}, {el.y:.0f}) conf={el.confidence:.2f}")
    except Exception as e:
        print(f"Could not scan: {e}")
        print("(This is expected if no matching elements are visible)")

    # Example: Click a menu item (uncomment to test)
    # nv.click("File")
    # time.sleep(0.5)
    # nv.click("New")

    # Example: Type into a field (uncomment to test)
    # nv.type_into("Search", "Hello World")

    # Example: Use relative positioning (uncomment to test)
    # nv.click("Delete", relative_to="Invoice #4920", direction="right")

    print("\nDemo complete!")


if __name__ == "__main__":
    main()
