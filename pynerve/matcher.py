"""Fuzzy text matcher for finding UI elements by text labels.

Confidence semantics:
    - ``Element.confidence`` is on a **0.0–1.0** scale (from the OCR or UIA engine).
    - ``threshold`` parameters are on a **0–100** scale (user-facing percentage).
    - The comparison ``el.confidence * 100 >= threshold`` bridges the two scales.
      This is intentional: the engine stores normalized floats, but human-facing
      thresholds are more intuitive as percentages (e.g. ``confidence=80`` means
      "at least 80% OCR certainty").
"""

from __future__ import annotations

import logging

from rapidfuzz import fuzz, process

from ._types import Element

logger = logging.getLogger("pynerve.matcher")


def find_match(
    target: str,
    elements: list[Element],
    threshold: int = 80,
) -> Element | None:
    """Find the best matching element for the target text.

    Uses token_set_ratio for partial/fuzzy matches, handling cases like:
    - "Settings" matching "Settings (Modified)"
    - Case-insensitive matching
    - Partial string matching

    Args:
        target: The text to search for.
        elements: List of Element objects to search through.
        threshold: Minimum similarity score (0-100) to consider a match.

    Returns:
        Best matching Element, or None if no match above threshold.
    """
    if not elements:
        return None

    target_lower = target.lower().strip()

    # Map index -> normalized text for all non-empty elements
    choices: dict[int, str] = {}
    for idx, el in enumerate(elements):
        norm = el.text.lower().strip()
        if norm:
            choices[idx] = norm

    if not choices:
        return None

    # Try exact match first (preserving top-to-bottom order).
    # Like every other branch, exact matches must still meet the OCR
    # confidence threshold — otherwise low-confidence noise would match
    # exactly while higher-quality fuzzy paths correctly reject it.
    for idx, norm in choices.items():
        if norm == target_lower:
            el = elements[idx]
            if el.confidence * 100 >= threshold:
                return el
            logger.debug(
                "Exact text match below OCR confidence threshold: '%s' (conf=%.2f)",
                el.text, el.confidence,
            )

    # Try contains match (preserving top-to-bottom order)
    # Phase 1: Super-string contains match (e.g. search for "Save" matches "Save Changes")
    for idx, norm in choices.items():
        if target_lower in norm:
            el = elements[idx]
            if el.confidence * 100 >= threshold:
                logger.debug("Super-string contains match: '%s' -> '%s' (conf=%.2f)", target, el.text, el.confidence)
                return el

    # Phase 2: Sub-string contains match (e.g. search for "Save Document" matches "Save")
    for idx, norm in choices.items():
        if norm in target_lower:
            el = elements[idx]
            if el.confidence * 100 >= threshold:
                logger.debug("Sub-string contains match: '%s' -> '%s' (conf=%.2f)", target, el.text, el.confidence)
                return el

    # Fuzzy match. For short words (<= 4 chars), use strict fuzz.ratio to prevent
    # over-matching like "edit" falsely matching "audit" (which token_set_ratio conflates).
    scorer = fuzz.ratio if len(target_lower) <= 4 else fuzz.token_set_ratio
    result = process.extractOne(
        target_lower,
        choices,
        scorer=scorer,
        score_cutoff=threshold,
    )

    if result is None:
        return None

    matched_text, score, idx = result
    element = elements[idx]
    if element.confidence * 100 < threshold:
        logger.debug(
            "Fuzzy match below OCR confidence threshold: '%s' -> '%s' (score=%.1f, conf=%.2f)",
            target, element.text, score, element.confidence,
        )
        return None

    logger.debug("Fuzzy match: '%s' -> '%s' (score=%.1f, conf=%.2f)", target, element.text, score, element.confidence)
    return element


def find_all_matches(
    target: str,
    elements: list[Element],
    threshold: int = 80,
    limit: int = 10,
) -> list[tuple[Element, float]]:
    """Find all matching elements above the threshold.

    Args:
        target: The text to search for.
        elements: List of Element objects to search through.
        threshold: Minimum similarity score (0-100).
        limit: Maximum number of results to return.

    Returns:
        List of (Element, score) tuples, sorted by match quality descending.
    """
    if not elements:
        return []

    target_lower = target.lower().strip()

    # Map index -> normalized text for all non-empty elements
    choices: dict[int, str] = {}
    for idx, el in enumerate(elements):
        norm = el.text.lower().strip()
        if norm:
            choices[idx] = norm

    if not choices:
        return []

    matches: list[tuple[Element, float]] = []
    seen_indices: set[int] = set()

    # Phase 1: Exact matches (Score 100.0)
    for idx, norm in choices.items():
        if norm == target_lower:
            el = elements[idx]
            if el.confidence * 100 >= threshold:
                matches.append((el, 100.0))
                seen_indices.add(idx)

    # Phase 2: Whole word / contains matches (Score 92.0 - 95.0)
    for idx, norm in choices.items():
        if idx in seen_indices:
            continue
        el = elements[idx]
        if el.confidence * 100 < threshold:
            continue

        words = norm.split()
        if target_lower in words:
            matches.append((el, 95.0))
            seen_indices.add(idx)
        elif target_lower in norm:
            matches.append((el, 90.0))
            seen_indices.add(idx)

    # Phase 3: Fuzzy matches for remaining items
    scorer = fuzz.ratio if len(target_lower) <= 4 else fuzz.token_set_ratio
    fuzzy_results = process.extract(
        target_lower,
        {idx: norm for idx, norm in choices.items() if idx not in seen_indices},
        scorer=scorer,
        score_cutoff=threshold,
        limit=limit,
    )

    for matched_text, score, idx in fuzzy_results:
        el = elements[idx]
        if el.confidence * 100 >= threshold:
            matches.append((el, float(score)))

    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[:limit]



def filter_by_direction(
    candidates: list[Element],
    anchor: Element,
    direction: str,
) -> Element | None:
    """Filter candidates by spatial direction relative to an anchor element.

    Args:
        candidates: List of candidate elements to filter.
        anchor: The reference element.
        direction: One of "right", "left", "above", "below".

    Returns:
        Closest matching element in the specified direction, or None.
    """
    ax, ay = anchor.center
    filtered: list[tuple[Element, float]] = []

    for el in candidates:
        ex, ey = el.center
        dx = ex - ax
        dy = ey - ay

        if direction == "right" and dx > 0 and abs(dy) <= abs(dx) * 1.5:
            filtered.append((el, abs(dx) + abs(dy) * 0.5))
        elif direction == "left" and dx < 0 and abs(dy) <= abs(dx) * 1.5:
            filtered.append((el, abs(dx) + abs(dy) * 0.5))
        elif direction == "above" and dy < 0 and abs(dx) <= abs(dy) * 1.5:
            filtered.append((el, abs(dy) + abs(dx) * 0.5))
        elif direction == "below" and dy > 0 and abs(dx) <= abs(dy) * 1.5:
            filtered.append((el, abs(dy) + abs(dx) * 0.5))

    if not filtered:
        return None

    # Return closest match
    filtered.sort(key=lambda x: x[1])
    return filtered[0][0]
