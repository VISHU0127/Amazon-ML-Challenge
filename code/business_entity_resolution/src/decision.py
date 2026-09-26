"""One decision implementation for production and paired diagnostic runs."""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from config import ARTIFACTS_DIR


def load_decision_params(path=None):
    path = Path(path or Path(ARTIFACTS_DIR) / 'decision_params.json')
    params = json.loads(path.read_text())
    required = {'optimal_threshold', 'global_uniqueness', 'confidence_gap',
                'confidence_ceiling', 'colocation_guard', 'street_number_gate',
                'name_floor', 'budget_caps'}
    if required - params.keys():
        raise ValueError(f"Stale/incomplete decision config {path}: missing {sorted(required - params.keys())}")
    for field, keys in {
        'street_number_gate': {'enabled', 'max_difference'},
        'name_floor': {'enabled', 'indic_pattern', 'indic_jaro_winkler_min',
                       'token_overlap_min', 'levenshtein_min', 'levenshtein_alternative_min'},
        'budget_caps': {'S2', 'S3', 'total'},
    }.items():
        if keys - params[field].keys():
            raise ValueError(f"Incomplete {field} in {path}")
    if not 0 <= params['optimal_threshold'] <= 1 or not 0 <= params['confidence_gap'] <= 1:
        raise ValueError('Invalid threshold or confidence gap')
    if params['street_number_gate']['max_difference'] < 0:
        raise ValueError('Street number tolerance must be nonnegative')
    if any(cap is not None and (not isinstance(cap, int) or cap < 1)
           for cap in params['budget_caps'].values()):
        raise ValueError('Budget caps must be positive integers or null')
    return params


def calibrated_probabilities(model, cal_info, matrix):
    raw = model.predict_proba(matrix)[:, 1]
    if cal_info['type'] == 'isotonic':
        return cal_info['calibrator'].predict(raw)
    if cal_info['type'] == 'platt':
        clipped = np.clip(raw, 1e-7, 1 - 1e-7)
        return cal_info['calibrator'].predict_proba(np.log(clipped / (1 - clipped)).reshape(-1, 1))[:, 1]
    raise ValueError(f"Unsupported calibration type: {cal_info['type']}")


def passes_pair_gates(features, source, target, params, probability):
    street = params['street_number_gate']
    sn1, sn2 = source['street_num'], target['street_num']
    if street['enabled'] and sn1.isdigit() and sn2.isdigit():
        if abs(int(sn1) - int(sn2)) > street['max_difference']:
            return False
    floor = params['name_floor']
    if floor['enabled']:
        if re.search(floor['indic_pattern'], source['raw_name'] + target['raw_name']):
            if features['name_jaro_winkler'] < floor['indic_jaro_winkler_min']:
                return False
        elif not ((features['name_token_overlap_ratio'] >= floor['token_overlap_min']
                   and features['name_levenshtein_ratio'] >= floor['levenshtein_min'])
                  or features['name_levenshtein_ratio'] >= floor['levenshtein_alternative_min']):
            return False
    if params['colocation_guard'] and probability < .85:
        if (features['addr_token_overlap_ratio'] >= .70 and features['name_jaro_winkler'] < .40
                and features['name_embedding_cosine'] < .40 and features['name_token_jaccard'] == 0):
            return False
    return True


def qualify_pairs(pairs, matrix, probabilities, source_cache, target_cache, params, feature_names):
    grouped = defaultdict(list)
    for i, (sid, tid, *_) in enumerate(pairs):
        grouped[sid].append((i, tid, float(probabilities[i])))
    accepted = []
    tau, gap = params['optimal_threshold'], params['confidence_gap']
    for sid, candidates in grouped.items():
        candidates.sort(key=lambda row: row[2], reverse=True)
        if gap > 0 and len(candidates) >= 2:
            top, second = candidates[0][2], candidates[1][2]
            if tau <= top < params['confidence_ceiling'] and top - second < gap:
                continue
        for i, tid, probability in candidates:
            if probability < tau:
                continue
            features = dict(zip(feature_names, matrix[i]))
            if passes_pair_gates(features, source_cache[sid], target_cache[tid], params, probability):
                accepted.append((sid, tid, probability))
    return accepted


def resolve_pairs(accepted, params):
    claimed = set()
    result = defaultdict(list)
    counts = defaultdict(lambda: defaultdict(int))
    caps = params['budget_caps']
    for sid, tid, probability in sorted(accepted, key=lambda row: row[2], reverse=True):
        source = tid.split('-')[0]
        if params['global_uniqueness'] and tid in claimed:
            continue
        if caps['total'] is not None and len(result[sid]) >= caps['total']:
            continue
        if caps[source] is not None and counts[sid][source] >= caps[source]:
            continue
        result[sid].append(tid)
        counts[sid][source] += 1
        claimed.add(tid)
    return result
