"""Candidate-pair feature extraction for Amazon ML Challenge 2026.

Uses only supplied TSVs. All string normalization and similarity computation
are strictly local with zero external web/API calls.
"""

from __future__ import annotations

import re
import unicodedata
from typing import NamedTuple
import rapidfuzz.fuzz as fuzz

TOKEN = re.compile(r"[a-z0-9]+")
NUMERIC = re.compile(r"^\d{1,6}$")
ALIASES = {
    "corp": "corporation", "co": "company", "pvt": "private",
    "ltd": "limited", "st": "street", "rd": "road", "ave": "avenue",
    "blvd": "boulevard", "ctr": "center", "dr": "drive", "ln": "lane",
    "ste": "suite", "apt": "apartment", "hwy": "highway", "fl": "floor"
}
LEGAL = {
    "inc", "incorporated", "llc", "corporation", "company", "private",
    "limited", "plc", "llp", "corp", "pvt", "ltd"
}
ADDRESS_GENERIC = {
    "street", "road", "avenue", "boulevard", "near", "opposite", "floor",
    "unit", "building", "block", "lane", "drive", "suite", "apartment",
    "circle", "court", "parkway", "highway"
}


def normalize_tokens(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    if value.isascii():
        ascii_text = value.lower().replace("&", " and ")
    else:
        normalized = unicodedata.normalize("NFKD", value.casefold().replace("&", " and "))
        ascii_text = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return tuple(ALIASES.get(token, token) for token in TOKEN.findall(ascii_text))


def clean_string(value: str) -> str:
    tokens = normalize_tokens(value)
    return " ".join(tokens)


def extract_legal_suffix(tokens: tuple[str, ...]) -> str | None:
    for token in reversed(tokens):
        if token in LEGAL:
            return token
    return None


def extract_house_number(tokens: tuple[str, ...]) -> str | None:
    return next((x for x in tokens if NUMERIC.fullmatch(x)), None)


def extract_numeric_tokens(tokens: tuple[str, ...]) -> set[str]:
    return {x for x in tokens if NUMERIC.fullmatch(x)}


class EntityProfile:
    __slots__ = (
        "raw_name", "raw_addr", "country",
        "name_tokens", "addr_tokens",
        "name_clean", "addr_clean",
        "name_set", "addr_set",
        "legal_suffix", "house_num", "numeric_tokens"
    )

    def __init__(self, name: str, addr: str, country: str):
        self.raw_name = name or ""
        self.raw_addr = addr or ""
        self.country = country.strip().casefold() if country else ""
        self.name_tokens = normalize_tokens(self.raw_name)
        self.addr_tokens = normalize_tokens(self.raw_addr)
        self.name_clean = " ".join(self.name_tokens)
        self.addr_clean = " ".join(self.addr_tokens)
        self.name_set = set(self.name_tokens)
        self.addr_set = set(self.addr_tokens)
        self.legal_suffix = extract_legal_suffix(self.name_tokens)
        self.house_num = extract_house_number(self.addr_tokens)
        self.numeric_tokens = extract_numeric_tokens(self.addr_tokens)


FEATURE_NAMES = [
    # Country (1)
    "country_match",
    # Name features (9)
    "name_exact",
    "name_jaccard",
    "name_overlap",
    "name_fuzz_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_len_diff",
    "name_token_count_diff",
    "name_legal_suffix_status",
    # Address features (10)
    "addr_exact",
    "addr_jaccard",
    "addr_overlap",
    "addr_fuzz_ratio",
    "addr_token_sort_ratio",
    "addr_len_diff",
    "addr_missing",
    "house_num_status",
    "numeric_jaccard",
    "numeric_conflict",
    # Provenance / Interaction features (6)
    "name_x_addr_jaccard",
    "name_x_addr_sort",
    "sketch_score",
    "reciprocal_rank",
    "is_top1",
    "total_candidates"
]


def extract_pair_features(
    s1: EntityProfile,
    s2: EntityProfile,
    rank: int,
    total_cands: int
) -> list[float]:
    """Compute dense numerical feature vector for candidate pair."""
    # Country agreement
    country_match = 1.0 if s1.country and s2.country and s1.country == s2.country else 0.0

    # Name features
    name_exact = 1.0 if s1.name_clean and s1.name_clean == s2.name_clean else 0.0
    name_union = s1.name_set | s2.name_set
    name_inter = s1.name_set & s2.name_set
    name_jaccard = len(name_inter) / len(name_union) if name_union else 0.0
    min_name_len = min(len(s1.name_set), len(s2.name_set))
    name_overlap = len(name_inter) / min_name_len if min_name_len else 0.0

    # RapidFuzz character/token metrics
    if s1.name_clean and s2.name_clean:
        name_fuzz_ratio = fuzz.ratio(s1.name_clean, s2.name_clean) / 100.0
        name_token_sort_ratio = fuzz.token_sort_ratio(s1.name_clean, s2.name_clean) / 100.0
        name_token_set_ratio = fuzz.token_set_ratio(s1.name_clean, s2.name_clean) / 100.0
    else:
        name_fuzz_ratio = name_token_sort_ratio = name_token_set_ratio = 0.0

    name_len_diff = float(abs(len(s1.name_clean) - len(s2.name_clean)))
    name_token_count_diff = float(abs(len(s1.name_tokens) - len(s2.name_tokens)))

    if s1.legal_suffix and s2.legal_suffix:
        name_legal_suffix_status = 1.0 if s1.legal_suffix == s2.legal_suffix else -1.0
    else:
        name_legal_suffix_status = 0.0

    # Address features
    addr_missing = 1.0 if not s1.addr_clean or not s2.addr_clean else 0.0
    addr_exact = 1.0 if s1.addr_clean and s1.addr_clean == s2.addr_clean else 0.0
    addr_union = s1.addr_set | s2.addr_set
    addr_inter = s1.addr_set & s2.addr_set
    addr_jaccard = len(addr_inter) / len(addr_union) if addr_union else 0.0
    min_addr_len = min(len(s1.addr_set), len(s2.addr_set))
    addr_overlap = len(addr_inter) / min_addr_len if min_addr_len else 0.0

    if s1.addr_clean and s2.addr_clean:
        addr_fuzz_ratio = fuzz.ratio(s1.addr_clean, s2.addr_clean) / 100.0
        addr_token_sort_ratio = fuzz.token_sort_ratio(s1.addr_clean, s2.addr_clean) / 100.0
    else:
        addr_fuzz_ratio = addr_token_sort_ratio = 0.0

    addr_len_diff = float(abs(len(s1.addr_clean) - len(s2.addr_clean)))

    # House number status: +1 match, -1 contradiction, 0 missing
    if s1.house_num and s2.house_num:
        house_num_status = 1.0 if s1.house_num == s2.house_num else -1.0
    else:
        house_num_status = 0.0

    # Numeric tokens (pincodes, street numbers)
    num_union = s1.numeric_tokens | s2.numeric_tokens
    num_inter = s1.numeric_tokens & s2.numeric_tokens
    numeric_jaccard = len(num_inter) / len(num_union) if num_union else 0.0
    numeric_conflict = 1.0 if (s1.numeric_tokens and s2.numeric_tokens and not num_inter) else 0.0

    # Interactions & provenance
    name_x_addr_jaccard = name_jaccard * addr_jaccard
    name_x_addr_sort = name_token_sort_ratio * (addr_token_sort_ratio if not addr_missing else 0.6)

    # Sketch preliminary score emulation
    sketch_score = 4.0 * name_jaccard + 2.0 * addr_jaccard
    if house_num_status == 1.0:
        sketch_score += 0.8
    elif house_num_status == -1.0:
        sketch_score -= 0.7
    if country_match:
        sketch_score += 0.25

    reciprocal_rank = 1.0 / rank if rank > 0 else 0.0
    is_top1 = 1.0 if rank == 1 else 0.0
    total_cands_f = float(total_cands)

    return [
        country_match,
        name_exact,
        name_jaccard,
        name_overlap,
        name_fuzz_ratio,
        name_token_sort_ratio,
        name_token_set_ratio,
        name_len_diff,
        name_token_count_diff,
        name_legal_suffix_status,
        addr_exact,
        addr_jaccard,
        addr_overlap,
        addr_fuzz_ratio,
        addr_token_sort_ratio,
        addr_len_diff,
        addr_missing,
        house_num_status,
        numeric_jaccard,
        numeric_conflict,
        name_x_addr_jaccard,
        name_x_addr_sort,
        sketch_score,
        reciprocal_rank,
        is_top1,
        total_cands_f
    ]
