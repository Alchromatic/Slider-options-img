#!/usr/bin/env python3
"""
library_normalize.py
====================

Cross-source normalization for the Reference Image Library so both sources
(NGA, Artvee) map into ONE shared set of filter values: era, genre (type),
artist, nationality. Used by the ingestion adapters (new rows) and the backfill
(existing rows).
"""

from __future__ import annotations

import re
from typing import List, Optional

# Ordered era buckets keyed by century. First matching wins.
_ERA_NAMES = {
    15: "15th century", 16: "16th century", 17: "17th century",
    18: "18th century", 19: "19th century", 20: "20th century",
    21: "21st century",
}

# Genre inference: (label, keywords). First match wins; order = priority.
_GENRE_RULES = [
    ("Religious", ["madonna", "christ", "saint ", "st. ", "crucifix", "nativity",
                   "virgin", " angel", "holy ", "biblical", "adoration",
                   "annunciation", "apostle", "prophet", "baptism", "pietà", "pieta"]),
    ("Mythological", ["venus", "apollo", "diana", "nymph", "cupid", "bacchus",
                      "mytholog", "muse", "psyche", "minerva"]),
    ("Portrait", ["portrait", "self-portrait", "self portrait", " sir ", " lady ",
                  "bust of", "head of", "countess", "duke", "duchess", "king ",
                  "queen ", "madame", "monsieur"]),
    ("Still Life", ["still life", "still-life", "flower", "fruit", "bouquet",
                    "vase", "roses", "basket of"]),
    ("Marine", ["shipwreck", "ship", " sea", "seascape", "harbor", "harbour",
                "boat", "marine", "coast", "fishing"]),
    ("Landscape", ["landscape", "view of", "mountain", "river", "valley",
                   "forest", "countryside", "field", "garden", "village",
                   "waterfall", "lake"]),
    ("Animal", ["horse", " dog", "birds", " bird", "cattle", "lion", "animal",
                "cat ", "hunt"]),
    ("Abstract", ["abstract", "composition no", "composition with", "untitled"]),
]


def era_bucket(year_start: Optional[int], year_end: Optional[int] = None) -> str:
    """Map a year (or year range) to a named era bucket."""
    y = year_start if year_start is not None else year_end
    if y is None:
        return "Unknown"
    if y < 1400:
        return "Pre-15th century"
    century = (y // 100) + 1
    return _ERA_NAMES.get(century, "Unknown")


def infer_genre(*texts: Optional[str]) -> str:
    """Infer a normalized genre/type from any available text (title/medium/tags)."""
    blob = " " + " ".join((t or "").lower() for t in texts) + " "
    for label, keywords in _GENRE_RULES:
        if any(k in blob for k in keywords):
            return label
    return "Other"


# Map common demonyms/adjectives to a canonical nationality label.
_NAT_CANON = {
    "american": "American", "british": "British", "english": "British",
    "scottish": "British", "welsh": "British", "irish": "Irish",
    "french": "French", "italian": "Italian", "dutch": "Dutch",
    "flemish": "Flemish", "german": "German", "spanish": "Spanish",
    "belgian": "Belgian", "swiss": "Swiss", "austrian": "Austrian",
    "russian": "Russian", "chinese": "Chinese", "japanese": "Japanese",
    "danish": "Danish", "swedish": "Swedish", "norwegian": "Norwegian",
    "polish": "Polish", "greek": "Greek", "mexican": "Mexican",
    "canadian": "Canadian", "portuguese": "Portuguese", "hungarian": "Hungarian",
    "czech": "Czech", "finnish": "Finnish",
}


def clean_nationality(raw: Optional[str]) -> Optional[str]:
    """Normalize a nationality string (first token, canonical casing)."""
    if not raw:
        return None
    s = raw.strip().strip(",;").lower()
    if not s or s in ("other", "anonymous", "unidentified", "nationality unknown"):
        return None
    # Take the first nationality word if a phrase like "French, 1840 - 1926".
    first = re.split(r"[,/;(]", s)[0].strip()
    token = first.split()[0] if first.split() else first
    return _NAT_CANON.get(token, first.title() if first else None)


# The set of genres surfaced as filter options (drives the UI dropdown order).
GENRE_OPTIONS = ["Portrait", "Landscape", "Still Life", "Religious",
                 "Mythological", "Marine", "Animal", "Abstract", "Other"]

ERA_OPTIONS = ["Pre-15th century", "15th century", "16th century", "17th century",
               "18th century", "19th century", "20th century", "21st century", "Unknown"]


__all__ = ["era_bucket", "infer_genre", "clean_nationality", "GENRE_OPTIONS", "ERA_OPTIONS"]
