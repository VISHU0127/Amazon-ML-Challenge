#!/usr/bin/env python3
"""
Stage 1 — Unit tests for normalize.py

Tests normalization on ~20 hand-picked noisy pairs from training data,
showing before/after transformations.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from normalize import (
    normalize_name, normalize_address, normalize_country,
    extract_legal_suffixes, extract_postal_code, extract_street_number,
)

# ═══════════════════════════════════════════════════════════════════════
# TEST CASES — hand-picked from training pair mining
# ═══════════════════════════════════════════════════════════════════════

NAME_TEST_CASES = [
    # (raw_input, expected_normalized)  — expected is approximate / substring check
    # 1. Case folding
    ("DAVIS FAMILY OFFICE", "davis family office"),
    # 2. Legal suffix removal: Inc
    ("3520 Main Road Realty Inc", "3520 main road realty"),
    # 3. Legal suffix removal: Pvt. Ltd.
    ("Team Air Pvt. Ltd.", "team air"),
    # 4. Legal suffix removal: Private Limited
    ("Clairvoyant Record Private Limited", "clairvoyant record"),
    # 5. Legal suffix removal: LLC
    ("Grand Connecticut LLC", "grand connecticut"),
    # 6. Legal suffix removal: Corporation
    ("Desert Society Incorporated", "desert society"),
    # 7. Diacritics stripping
    ("Clairvoyant Récord Private Ltd", "clairvoyant record"),
    # 8. URL/domain stripping
    ("TEAMAIR.COM", "teamair"),
    # 9. Brackets/parens removal
    ("Davis Family (Office)", "davis family office"),
    # 10. Square brackets removal
    ("MS [Consultancy]", "ms consultancy"),
    # 11. & → and
    ("Unique & Sons Private Limited", "unique and sons"),
    # 12. Extra whitespace collapse
    ("Drayex  Berto LLC", "drayex berto"),
    # 13. Mixed: case + suffix + parens
    ("Clairvoyant Rceoad Private (Limited)", "clairvoyant rceoad"),
    # 14. Hindi suffixes — प्राइवेट (private) and लिमिटेड (limited) are stripped
    # After suffix removal, core name is preserved in Devanagari
    ("सुप्रीम आईटी प्राइवेट लिमिटेड", "सुप्रीम आईटी"),
    # 15. French legal form
    ("Marina Ecole France Sarl", "marina ecole france"),
    # 16. Multiple suffixes: PLLC + Partners
    ("Prairie Capital Partners PLLC", "prairie capital"),
    # 17. www prefix + .com
    ("www.cardiology.com", "cardiology"),
    # 18. Possessive 's
    ("Moyna's Coffee", "moyna coffee"),
    # 19. Leading punctuation / asterisks
    ("*** Royal  Packaging", "royal packaging"),
    # 20. DBA prefix
    ("Drexsolpyra DBA: Cornerstone Investments L.L.C.", "drexsolpyra cornerstone investments"),
]

ADDRESS_TEST_CASES = [
    # (raw_input, expected_contains_these_tokens)
    # 1. Circle → cir
    ("88 Olive Circle, Lebanon, TN", ["88", "olive", "cir", "lebanon", "tn"]),
    # 2. Boulevard → blvd
    ("120 Autumn Woods Boulevard, Mount Holly, NC", ["120", "autumn", "woods", "blvd", "mount", "holly", "nc"]),
    # 3. Road → rd
    ("4038 Talmadge Road, Unit 102, Toledo, OH", ["4038", "talmadge", "rd", "unit", "102", "toledo", "oh"]),
    # 4. Avenue → ave
    ("351 Hoxie Avenue, Calumet City, IL", ["351", "hoxie", "ave", "calumet", "city", "il"]),
    # 5. Maharashtra → mh
    ("Pune, Maharashtra", ["pune", "mh"]),
    # 6. West Bengal → wb (multi-word)
    ("Howrah, West Bengal", ["howrah", "wb"]),
    # 7. Texas → tx
    ("158 Simpson Lane, Somerset, Texas", ["158", "simpson", "ln", "somerset", "tx"]),
    # 8. Drive → dr
    ("19821 Wheelwright Drive, Montgomery Village, MD",
     ["19821", "wheelwright", "dr", "montgomery", "village", "md"]),
    # 9. Street → st (kentucky is also a US state → ky)
    ("67 KENTUCKY ST, SALYERSVILLE, KY", ["67", "ky", "st", "salyersville", "ky"]),
    # 10. Diacritics in address
    ("175 Boulevard du Président Franklin Roosevelt, Bordeaux",
     ["175", "blvd", "du", "president", "franklin", "roosevelt", "bordeaux"]),
    # 11. N/A removal
    ("CALUMET CITY, 351 HOXIE AVE, N/A, IL", ["calumet", "city", "351", "hoxie", "ave", "il"]),
    # 12. ## prefix on numbers
    ("##19821 WHEELWRIGHT DR, MONTGOMERY VILLAGE, MD",
     ["19821", "wheelwright", "dr", "montgomery", "village", "md"]),
    # 13. Hindi state name
    ("Pune, महाराष्ट्र", ["pune", "mh"]),
    # 14. Madhya Pradesh → mp
    ("Sehore, Madhya Pradesh", ["sehore", "mp"]),
    # 15. Empty address
    ("", []),
]

POSTAL_CODE_CASES = [
    ("88 Olive Circle, Lebanon, TN 37090", "37090"),
    ("FLAT NO:101, HYDERABAD, Telangana 500004", "500004"),
    ("63 R. DE DIEPPE, LILLE 59000", "59000"),
    ("No postal code here", ""),
]

STREET_NUMBER_CASES = [
    ("88 Olive Circle, Lebanon, TN", "88"),
    ("##19821 WHEELWRIGHT DR", "19821"),
    ("#4038 TALMADGE RD", "4038"),
    ("9/1/3 Kasundia 2Nd Bye Lane", "9/1/3"),
    ("No number here", ""),
]

COUNTRY_CASES = [
    ("US", "us"),
    ("India", "india"),
    ("France", "france"),
    ("  US  ", "us"),
]


# ═══════════════════════════════════════════════════════════════════════
# RUN TESTS
# ═══════════════════════════════════════════════════════════════════════

def run_tests():
    passed = 0
    failed = 0
    total = 0

    print("=" * 70)
    print("NORMALIZATION UNIT TESTS")
    print("=" * 70)

    # ── Name tests ──────────────────────────────────────────────────────
    print("\n── Name normalization ──")
    for i, (raw, expected) in enumerate(NAME_TEST_CASES, 1):
        total += 1
        result = normalize_name(raw)
        ok = result == expected
        status = "✓" if ok else "✗"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  {status} {i:2d}. [{raw}]")
        print(f"       → [{result}]")
        if not ok:
            print(f"       EXPECTED: [{expected}]")
    # Also show suffix extraction
    print("\n── Legal suffix extraction ──")
    suffix_tests = [
        ("Team Air Pvt. Ltd.", ["ltd", "pvt"]),
        ("Grand Connecticut LLC", ["llc"]),
        ("Marina Ecole France Sarl", ["sarl"]),
        ("सुप्रीम आईटी प्राइवेट लिमिटेड", ["ltd", "pvt"]),
    ]
    for raw, expected_suffixes in suffix_tests:
        total += 1
        result = extract_legal_suffixes(raw)
        ok = result == expected_suffixes
        status = "✓" if ok else "✗"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  {status} [{raw}] → suffixes: {result}")
        if not ok:
            print(f"       EXPECTED: {expected_suffixes}")

    # ── Address tests ───────────────────────────────────────────────────
    print("\n── Address normalization ──")
    for i, (raw, expected_tokens) in enumerate(ADDRESS_TEST_CASES, 1):
        total += 1
        result = normalize_address(raw)
        result_tokens = result.lower().split()
        # Check all expected tokens are present
        missing = [t for t in expected_tokens if t not in result_tokens]
        ok = len(missing) == 0
        status = "✓" if ok else "✗"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  {status} {i:2d}. [{raw}]")
        print(f"       → [{result}]")
        if missing:
            print(f"       MISSING TOKENS: {missing}")

    # ── Postal code tests ──────────────────────────────────────────────
    print("\n── Postal code extraction ──")
    for raw, expected in POSTAL_CODE_CASES:
        total += 1
        result = extract_postal_code(raw)
        ok = result == expected
        status = "✓" if ok else "✗"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  {status} [{raw}] → [{result}]")
        if not ok:
            print(f"       EXPECTED: [{expected}]")

    # ── Street number tests ────────────────────────────────────────────
    print("\n── Street number extraction ──")
    for raw, expected in STREET_NUMBER_CASES:
        total += 1
        result = extract_street_number(raw)
        ok = result == expected
        status = "✓" if ok else "✗"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  {status} [{raw}] → [{result}]")
        if not ok:
            print(f"       EXPECTED: [{expected}]")

    # ── Country tests ──────────────────────────────────────────────────
    print("\n── Country normalization ──")
    for raw, expected in COUNTRY_CASES:
        total += 1
        result = normalize_country(raw)
        ok = result == expected
        status = "✓" if ok else "✗"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  {status} [{raw}] → [{result}]")
        if not ok:
            print(f"       EXPECTED: [{expected}]")

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"RESULTS: {passed}/{total} passed, {failed} failed")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
