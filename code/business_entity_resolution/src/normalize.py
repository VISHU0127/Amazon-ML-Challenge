#!/usr/bin/env python3
"""
Stage 1 — Text Normalization for Business Entity Resolution

Provides normalize_name() and normalize_address() that clean noisy
business records into a canonical form suitable for similarity comparison.

All canonicalization tables are derived from training-data patterns or
standard abbreviation knowledge — no external data lookups.
"""

import re
import unicodedata


# ═══════════════════════════════════════════════════════════════════════
# LEGAL-SUFFIX CANONICALIZATION TABLE
# Mined from training name pairs: map every observed variant to a
# canonical short form.  During normalization we *strip* these entirely
# (they are noise for matching) but also expose an API to *extract* the
# canonical suffix if needed as a feature.
# ═══════════════════════════════════════════════════════════════════════

# Maps every observed surface form (lower-cased) → canonical suffix tag.
LEGAL_SUFFIX_MAP = {
    # Incorporated
    "inc": "inc", "inc.": "inc", "incorporated": "inc",
    # Corporation
    "corp": "corp", "corp.": "corp", "corporation": "corp",
    # Company
    "co": "co", "co.": "co", "company": "co",
    # Limited
    "ltd": "ltd", "ltd.": "ltd", "limited": "ltd",
    # Private
    "pvt": "pvt", "pvt.": "pvt", "private": "pvt",
    # LLC / LLP / PLLC
    "llc": "llc", "l.l.c.": "llc", "l.l.c": "llc",
    "llp": "llp", "l.l.p.": "llp", "l.l.p": "llp",
    "pllc": "pllc",
    # PLC
    "plc": "plc", "p.l.c.": "plc",
    # Public
    "public": "public",
    # Group / Holdings / Partners / Partnership
    "group": "group", "grp": "group",
    "holdings": "holdings", "holding": "holdings",
    "partners": "partners", "partnership": "partners",
    # Associates / Association
    "associates": "assoc", "association": "assoc", "assoc": "assoc",
    "assoc.": "assoc",
    # Enterprises / Enterprise
    "enterprises": "enterprise", "enterprise": "enterprise",
    # Foundation
    "foundation": "foundation", "fdn": "foundation",
    # Services / Service
    "services": "services", "service": "services",
    # Solutions / Solution
    "solutions": "solutions", "solution": "solutions",
    # Industries / Industrial
    "industries": "industries", "industrial": "industries",
    # Technologies / Technology / Tech
    "technologies": "tech", "technology": "tech", "tech": "tech",
    # Consulting / Consultants / Consultant
    "consulting": "consulting", "consultants": "consulting",
    "consultant": "consulting",
    # International / Global / Worldwide / National
    "international": "intl", "intl": "intl", "intl.": "intl",
    "global": "global", "worldwide": "worldwide", "national": "national",
    # Manufacturing / Mfg
    "manufacturing": "mfg", "mfg": "mfg", "mfg.": "mfg",
    # Trading / Tr
    "trading": "trading", "tr": "trading",
    # DBA
    "dba": "dba", "dba:": "dba",
    # French legal forms
    "sarl": "sarl", "s.a.r.l.": "sarl", "s.a.r.l": "sarl",
    "sas": "sas", "s.a.s.": "sas", "s.a.s": "sas",
    "sasu": "sasu",
    "sa": "sa", "s.a.": "sa", "s.a": "sa",
    "eurl": "eurl", "e.u.r.l.": "eurl",
    "sci": "sci", "s.c.i.": "sci",
    "snc": "snc", "s.n.c.": "snc",
    "gie": "gie",
    "scop": "scop",
    # German
    "gmbh": "gmbh", "ag": "ag",
    # Hindi equivalents
    "प्राइवेट": "pvt",
    "लिमिटेड": "ltd",
    "प्रा": "pvt",
    "प्रा.": "pvt",
    "लि": "ltd",
    "लि.": "ltd",
    "एलएलपी": "llp",
}

# Regex to match legal suffixes at word boundaries (case-insensitive)
_LEGAL_SUFFIX_PATTERN = re.compile(
    r'\b(' + '|'.join(
        re.escape(k) for k in sorted(LEGAL_SUFFIX_MAP.keys(), key=len, reverse=True)
    ) + r')\.?\b',
    re.IGNORECASE
)


# ═══════════════════════════════════════════════════════════════════════
# ADDRESS ABBREVIATION TABLE
# Mined from training address pairs: map long forms → short canonical.
# ═══════════════════════════════════════════════════════════════════════

ADDRESS_ABBREV_MAP = {
    # Street types
    "street": "st", "st.": "st",
    "road": "rd", "rd.": "rd",
    "avenue": "ave", "ave.": "ave",
    "boulevard": "blvd", "blvd.": "blvd",
    "drive": "dr", "dr.": "dr",
    "lane": "ln", "ln.": "ln",
    "court": "ct", "ct.": "ct",
    "circle": "cir", "cir.": "cir",
    "place": "pl", "pl.": "pl",
    "way": "way",
    "terrace": "ter", "ter.": "ter",
    "trail": "trl", "trl.": "trl",
    "highway": "hwy", "hwy.": "hwy",
    "parkway": "pkwy", "pkwy.": "pkwy",
    "expressway": "expy", "expy.": "expy",
    "pike": "pike",
    "route": "rte", "rte.": "rte",
    "square": "sq", "sq.": "sq",
    # Directionals
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw",
    "southeast": "se", "southwest": "sw",
    # Building / unit
    "apartment": "apt", "apt.": "apt",
    "suite": "ste", "ste.": "ste",
    "building": "bldg", "bldg.": "bldg",
    "floor": "fl", "fl.": "fl",
    "number": "no", "no.": "no",
    "plot": "plot",
    "house": "house",
    "flat": "flat",
    "block": "blk", "blk.": "blk",
    "unit": "unit",
    "door": "door",
    "sector": "sec", "sec.": "sec",
    # Indian states (full → abbreviation)
    "maharashtra": "mh",
    "karnataka": "ka",
    "telangana": "tg",
    "tamil nadu": "tn",
    "west bengal": "wb",
    "uttar pradesh": "up",
    "madhya pradesh": "mp",
    "andhra pradesh": "ap",
    "rajasthan": "rj",
    "gujarat": "gj",
    "kerala": "kl",
    "haryana": "hr",
    "punjab": "pb",
    "bihar": "br",
    "odisha": "od",
    "jharkhand": "jh",
    "chhattisgarh": "cg",
    "assam": "as",
    "uttarakhand": "uk",
    "himachal pradesh": "hp",
    "goa": "ga",
    "delhi": "dl",
    "new delhi": "dl",
    # Hindi state names → abbreviation
    "महाराष्ट्र": "mh",
    "कर्नाटक": "ka",
    "तेलंगाना": "tg",
    "తెలంగాణ": "tg",
    "तमिल नाडु": "tn",
    "पश्चिम बंगाल": "wb",
    "পশ্চিমবঙ্গ": "wb",
    "उत्तर प्रदेश": "up",
    "मध्य प्रदेश": "mp",
    "राजस्थान": "rj",
    "गुजरात": "gj",
    "केरल": "kl",
    # US state full → abbreviation
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct",
    "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi",
    "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me",
    "maryland": "md", "massachusetts": "ma", "michigan": "mi",
    "minnesota": "mn", "mississippi": "ms", "missouri": "mo",
    "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd",
    "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd",
    "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc",
    # French regions (common ones seen in test data)
    "ile-de-france": "idf",
    "nouvelle-aquitaine": "naq",
    "hauts-de-france": "hdf",
    "occitanie": "occ",
    "provence-alpes-cote d'azur": "paca",
    "auvergne-rhone-alpes": "ara",
    "bretagne": "bre",
    "normandie": "nor",
    "pays de la loire": "pdl",
    "grand est": "ges",
    "bourgogne-franche-comte": "bfc",
    "centre-val de loire": "cvl",
}

# Build a regex for multi-word address abbreviations first, then single-word
_ADDR_MULTI_WORD = {k: v for k, v in ADDRESS_ABBREV_MAP.items() if " " in k or len(k) > 3}
_ADDR_MULTI_PATTERN = re.compile(
    r'\b(' + '|'.join(
        re.escape(k) for k in sorted(_ADDR_MULTI_WORD.keys(), key=len, reverse=True)
    ) + r')\b',
    re.IGNORECASE
)

_ADDR_SINGLE_WORD = {k: v for k, v in ADDRESS_ABBREV_MAP.items()
                     if " " not in k and k not in ("n", "s", "e", "w")}
_ADDR_SINGLE_PATTERN = re.compile(
    r'\b(' + '|'.join(
        re.escape(k) for k in sorted(_ADDR_SINGLE_WORD.keys(), key=len, reverse=True)
    ) + r')\.?\b',
    re.IGNORECASE
)


# ═══════════════════════════════════════════════════════════════════════
# POSTAL CODE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

# US ZIP codes (5 or 5+4)
_US_ZIP_RE = re.compile(r'\b(\d{5})(?:-\d{4})?\b')

# Indian PIN codes (6 digits, typically starting with 1-9)
_INDIA_PIN_RE = re.compile(r'\b([1-9]\d{5})\b')

# French postal codes (5 digits, typically starting with 0-9)
_FRANCE_ZIP_RE = re.compile(r'\b(\d{5})\b')

# Generic: any 5-6 digit code
_GENERIC_POSTAL_RE = re.compile(r'\b(\d{5,6})\b')


def extract_postal_code(address: str) -> str:
    """Extract the first plausible postal/PIN/ZIP code from an address string."""
    m = _GENERIC_POSTAL_RE.search(address)
    return m.group(1) if m else ""


# ═══════════════════════════════════════════════════════════════════════
# STREET NUMBER EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

_STREET_NUM_RE = re.compile(r'(?:^|[,\s])#*(\d+(?:[/\-]\d+)*)\b')


def extract_street_number(address: str) -> str:
    """Extract the leading street/house/plot number from an address."""
    m = _STREET_NUM_RE.search(address)
    return m.group(1) if m else ""


# ═══════════════════════════════════════════════════════════════════════
# CORE NORMALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def _unicode_normalize(text: str) -> str:
    """NFKD normalize and strip combining diacritics from Latin chars only.

    We only strip combining marks (category Mn) that follow a Latin base
    character.  This preserves Devanagari, Bengali, Telugu, etc. scripts
    whose combining marks are essential to the character.
    """
    nfkd = unicodedata.normalize("NFKD", text)
    out = []
    prev_latin = False
    for c in nfkd:
        cat = unicodedata.category(c)
        if cat == "Mn":
            # Drop combining mark only if the preceding base char was Latin
            if prev_latin:
                continue
        # Track whether this character is a Latin letter
        if cat.startswith("L"):
            # Latin letters are in the Basic Latin or Latin Extended blocks
            cp = ord(c)
            prev_latin = (cp < 0x0250) or (0x1E00 <= cp <= 0x1EFF)
        else:
            prev_latin = False
        out.append(c)
    return "".join(out)


def _strip_punctuation(text: str) -> str:
    """Remove punctuation except hyphens in compound words and periods in abbreviations."""
    # Remove brackets, parens, asterisks, quotes, pipes, etc.
    text = re.sub(r'[\[\](){}*"|`~!@#$%^&+=<>]', ' ', text)
    # Remove leading/trailing hyphens and dots, keep internal ones
    text = re.sub(r'(?<!\w)[.\-]|[.\-](?!\w)', ' ', text)
    # Collapse separators
    text = re.sub(r'[,;:]+', ' ', text)
    # Remove possessive 's
    text = re.sub(r"'s\b", "", text)
    # Remove remaining apostrophes
    text = text.replace("'", "")
    return text


def _collapse_whitespace(text: str) -> str:
    """Collapse multiple spaces into one and strip."""
    return re.sub(r'\s+', ' ', text).strip()


def _strip_url_domain(text: str) -> str:
    """Remove .com, .org, .net, www. from business names."""
    text = re.sub(r'\bwww\.', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\.(com|org|net|co\.in|in|fr)\b', '', text, flags=re.IGNORECASE)
    return text


def _apply_legal_suffix_removal(text: str) -> str:
    """Remove all legal suffixes from a business name."""
    return _LEGAL_SUFFIX_PATTERN.sub(' ', text)


def extract_legal_suffixes(name: str) -> list:
    """Return a sorted list of canonical legal suffixes found in a name."""
    found = set()
    for m in _LEGAL_SUFFIX_PATTERN.finditer(name.lower()):
        raw = m.group().strip('.')
        canonical = LEGAL_SUFFIX_MAP.get(raw.lower(), raw.lower())
        found.add(canonical)
    return sorted(found)


def normalize_name(name: str, keep_suffixes: bool = False) -> str:
    """
    Normalize a business name for comparison.

    Steps:
    1. Lowercase
    2. Unicode NFKD + strip diacritics
    3. Strip URLs/domains
    4. Remove (or keep) legal suffixes
    5. Strip punctuation
    6. Collapse whitespace

    Parameters
    ----------
    name : str
        Raw business name.
    keep_suffixes : bool
        If True, legal suffixes are canonicalized but kept; if False (default),
        they are stripped entirely for core-name comparison.

    Returns
    -------
    str
        Normalized name string.
    """
    if not name or not name.strip():
        return ""
    text = name.lower()
    text = _unicode_normalize(text)
    text = _strip_url_domain(text)
    if not keep_suffixes:
        text = _apply_legal_suffix_removal(text)
    # Normalize & → and BEFORE stripping punctuation (& is punctuation)
    text = text.replace("&", " and ")
    text = _strip_punctuation(text)
    text = _collapse_whitespace(text)
    return text


def normalize_address(address: str) -> str:
    """
    Normalize a business address for comparison.

    Steps:
    1. Lowercase
    2. Unicode NFKD + strip diacritics
    3. Apply multi-word abbreviations (state names first)
    4. Apply single-word abbreviations (street types)
    5. Strip punctuation
    6. Collapse whitespace

    Returns
    -------
    str
        Normalized address string.
    """
    if not address or not address.strip():
        return ""
    text = address.lower()
    # Apply multi-word abbreviations FIRST, before NFKD (includes Hindi/Bengali
    # state names that would be damaged by diacritics stripping)
    text = _ADDR_MULTI_PATTERN.sub(lambda m: ADDRESS_ABBREV_MAP.get(m.group().lower(), m.group()), text)
    # Apply single-word abbreviations (includes non-Latin single-word state names)
    text = _ADDR_SINGLE_PATTERN.sub(
        lambda m: ADDRESS_ABBREV_MAP.get(m.group().lower().rstrip('.'), m.group()), text
    )
    # NOW apply NFKD (safe since non-Latin abbreviations are already resolved)
    text = _unicode_normalize(text)
    text = _strip_punctuation(text)
    # Remove common noise prefixes
    text = re.sub(r'\b(h\.?no|door\s*no|plot\s*no|flat\s*no|kh\s*no|house\s*no)\b',
                  '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bn/?a\b', '', text)  # N/A
    text = _collapse_whitespace(text)
    return text


def normalize_country(country: str) -> str:
    """Normalize country to a canonical lowercase string (open vocabulary)."""
    if not country or not country.strip():
        return ""
    return country.strip().lower()
