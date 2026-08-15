from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Element:
    """A detected UI element from OCR or accessibility processing.

    Core fields (always present):
        text: Visible text content of the element.
        confidence: Detection confidence (0.0–1.0). OCR engines return a real
            confidence; the accessibility backend always returns 1.0.
        center: (x, y) pixel coordinates of the element center.
        bounds: (left, top, right, bottom) bounding box in pixels.

    Accessibility-enriched fields (None when using the vision backend):
        control_type: UIA control type (e.g. "Button", "Edit", "CheckBox").
        is_enabled: Whether the element is interactive / not grayed out.
        value: Current text value of editable controls (e.g. text fields).
    """

    text: str
    confidence: float
    center: tuple[float, float]
    bounds: tuple[float, float, float, float]
    # Accessibility-enriched fields (default None for vision backend)
    control_type: str | None = None
    is_enabled: bool | None = None
    value: str | None = None

    @property
    def x(self) -> float:
        return self.center[0]

    @property
    def y(self) -> float:
        return self.center[1]

    @property
    def width(self) -> float:
        return self.bounds[2] - self.bounds[0]

    @property
    def height(self) -> float:
        return self.bounds[3] - self.bounds[1]
