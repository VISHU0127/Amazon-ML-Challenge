"""
Central configuration for the Business Entity Resolution pipeline.
All paths are relative to PROJECT_ROOT so nothing is hardcoded.
"""

import os

# Root of the student_resource directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Dataset paths
TRAIN_DIR = os.path.join(PROJECT_ROOT, "dataset", "train")
TEST_DIR = os.path.join(PROJECT_ROOT, "dataset", "test")

TRAIN_SOURCE1 = os.path.join(TRAIN_DIR, "train_source1.tsv")
TRAIN_SOURCE2 = os.path.join(TRAIN_DIR, "train_source2.tsv")
TRAIN_SOURCE3 = os.path.join(TRAIN_DIR, "train_source3.tsv")
TRAIN_GROUND_TRUTH = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

TEST_SOURCE1 = os.path.join(TEST_DIR, "test_source1.tsv")
TEST_SOURCE2 = os.path.join(TEST_DIR, "test_source2.tsv")
TEST_SOURCE3 = os.path.join(TEST_DIR, "test_source3.tsv")

# Output paths
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
MATCHING_RESULTS = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_PAIRS = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

# Artifact paths
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")

# Validation split
VAL_SPLIT_PATH = os.path.join(ARTIFACTS_DIR, "val_split.json")

# Utility paths
VALIDATE_SCRIPT = os.path.join(PROJECT_ROOT, "utils", "validate_submission.py")

# Column names
COL_ENTITY_ID = "entity_id"
COL_BUSINESS_NAME = "business_name"
COL_BUSINESS_ADDRESS = "business_address"
COL_COUNTRY = "country"
COL_S1_ENTITY_ID = "source1_entity_id"
COL_MATCHED_IDS = "matched_entity_ids"
COL_CANDIDATE_IDS = "candidate_entity_ids"

# Reproducibility
RANDOM_SEED = 42
VAL_FRACTION = 0.15
