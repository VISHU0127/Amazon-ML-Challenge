"""Decision-only sweep. Requires saved probabilities; never loads an ML model.

Floor definitions are supplied as explicit historical AND/OR threshold terms.
Missing variants are rejected rather than guessed. This script writes reports,
not the active lock; a lock update requires examining the completed grid.
"""
import argparse
import copy
import csv
import hashlib
import json
import pickle
import re
import signal
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from config import ARTIFACTS_DIR
from decision import load_decision_params, qualify_pairs, resolve_pairs
from diagnose_embedding_fix import metrics
from features import FEATURE_NAMES

TAUS = [.30, .40, .50, .60, .65, .70, .75, .80, .85]


def timeout_handler(signum, frame):
    raise TimeoutError('Hard stop reached; no further sweep or lock changes')


def floor_mask(matrix, indic, definition):
    if not isinstance(definition, dict) or not definition.get('provenance'):
        raise ValueError('Exact historical floor definition/provenance is required')
    terms = definition.get('non_indic_any_of')
    if not terms:
        raise ValueError('Missing historical non-Indic floor terms')
    accepted = np.zeros(len(matrix), dtype=bool)
    for conjunction in terms:
        if not conjunction:
            raise ValueError('Empty floor conjunction is not permitted')
        term = np.ones(len(matrix), dtype=bool)
        for feature, threshold in conjunction.items():
            if feature not in FEATURE_NAMES or not isinstance(threshold, (int, float)):
                raise ValueError('Invalid historical floor condition')
            term &= matrix[:, FEATURE_NAMES.index(feature)] >= threshold
        accepted |= term
    jw_min = definition.get('indic_jaro_winkler_min')
    if not isinstance(jw_min, (int, float)):
        raise ValueError('Exact historical Indic branch is required')
    return np.where(indic, matrix[:, FEATURE_NAMES.index('name_jaro_winkler')] >= jw_min, accepted)


def run(args):
    remaining = datetime.fromisoformat(args.deadline_utc).timestamp() - time.time()
    if remaining <= 0:
        raise TimeoutError('The requested hard stop has already passed')
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(ARTIFACTS_DIR) / 'embedding_fix'
    report = json.loads((base / 'diagnostic_results.json').read_text())
    definitions = json.loads(Path(args.floors).read_text())
    for variant in ('overlap_and_lev',):
        if not definitions.get(variant):
            raise ValueError(f'Missing exact historical definition: {variant}')
    with open(base / 'full_pool_pairs.pkl', 'rb') as handle:
        data = pickle.load(handle)
    source, target, truth, pairs = (data[k] for k in ('source', 'target', 'truth', 'pairs'))
    saved = np.load(base / 'real_features.npz')
    matrix = saved['X']
    pair_hash = hashlib.sha256('\n'.join(f'{s}\t{t}\t{score}\t{rank}' for s,t,score,rank in pairs).encode()).hexdigest()
    sample_hash = hashlib.sha256('\n'.join(sorted(source)).encode()).hexdigest()
    if pair_hash != report['pair_sha256'] or str(saved['pair_sha256']) != pair_hash:
        raise ValueError('Candidate order differs from the trusted diagnostic')
    if sample_hash != report['sample_sha256'] or len(source) != 5000:
        raise ValueError('Source sample differs from the trusted diagnostic')
    if list(saved['feature_names']) != FEATURE_NAMES:
        raise ValueError('Feature order mismatch')
    with np.load(args.scores) as cache:
        required = {'probabilities', 'pair_sha256', 'model_sha256', 'calibrator_sha256'}
        if required - set(cache.files):
            raise ValueError(f'Score cache missing fields: {sorted(required-set(cache.files))}')
        for key in required - {'probabilities'}:
            if str(cache[key]) != report[key]:
                raise ValueError(f'Score cache provenance mismatch: {key}')
        probabilities = cache['probabilities']
    if probabilities.shape != (len(pairs),) or not np.isfinite(probabilities).all():
        raise ValueError('Invalid cached per-pair probabilities')
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError('Cached values are not calibrated probabilities')
    params = load_decision_params()
    if params != report['locked_config']:
        raise ValueError('Active config has changed since the trusted diagnostic')
    # The sole metric evaluator for the baseline and every sweep row.
    example = metrics(['x'], {'x': {'a','b'}}, {'x':['a','b','c']})
    assert abs(example['macro_f05'] - 5/7) < 1e-12
    baseline_pred = resolve_pairs(qualify_pairs(pairs, matrix, probabilities, source, target, params, FEATURE_NAMES), params)
    baseline = metrics(source, truth, baseline_pred)
    if abs(baseline['macro_f05'] - report['results']['locked_real']['macro_f05']) > 1e-12:
        raise ValueError('Cached scores do not reproduce the trusted 0.608125 baseline')

    regex = re.compile(params['name_floor']['indic_pattern'])
    indic = np.array([bool(regex.search(source[s]['raw_name'] + target[t]['raw_name'])) for s,t,*_ in pairs])
    floors = {'overlap_and_lev':floor_mask(matrix, indic, definitions['overlap_and_lev']),
              'no-floor':np.ones(len(matrix),dtype=bool)}
    streets = {tol:np.ones(len(pairs), dtype=bool) for tol in (0,1)}
    grouped = defaultdict(list)
    for i, (sid, tid, *_) in enumerate(pairs):
        grouped[sid].append(i)
        a,b = source[sid]['street_num'], target[tid]['street_num']
        if a.isdigit() and b.isdigit():
            difference = abs(int(a)-int(b))
            for tol in streets:
                streets[tol][i] = difference <= tol
    # Preserve existing tie ordering: per-source probability order, then stable
    # global probability order. Caps and uniqueness remain the production logic.
    ordered = {sid:sorted(indices, key=lambda i:float(probabilities[i]), reverse=True)
               for sid,indices in grouped.items()}
    streets['no-gate'] = np.ones(len(pairs),dtype=bool)
    rows = []
    fields = ['tau','floor','street_tolerance','macro_f05','pooled_precision',
              'pooled_recall','singleton_accuracy','macro_precision','macro_recall',
              'tp','fp','fn','singleton_correct','singleton_count']
    with (out / 'grid.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for tau in TAUS:
            for variant in ('overlap_and_lev', 'no-floor'):
                for tol in (0,1,'no-gate'):
                    if time.time() >= datetime.fromisoformat(args.deadline_utc).timestamp():
                        raise TimeoutError('Hard stop reached')
                    mask = floors[variant] & streets[tol] & (probabilities >= tau)
                    accepted = []
                    for sid, indices in ordered.items():
                        if len(indices) >= 2 and params['confidence_gap'] > 0:
                            top, second = (float(probabilities[i]) for i in indices[:2])
                            if tau <= top < params['confidence_ceiling'] and top-second < params['confidence_gap']:
                                continue
                        accepted.extend((sid, pairs[i][1], float(probabilities[i])) for i in indices if mask[i])
                    predictions = resolve_pairs(accepted, params)
                    result = metrics(source, truth, predictions)
                    row = {'tau':tau, 'floor':variant, 'street_tolerance':tol, **result}
                    row.pop('pooled_f05')
                    rows.append(row)
                    writer.writerow(row)
                    handle.flush()
                    print(json.dumps(row), flush=True)
    locked = next(r for r in rows if (r['tau'],r['floor'],r['street_tolerance']) == (.60,'overlap_and_lev',1))
    if abs(locked['macro_f05'] - baseline['macro_f05']) > 1e-12:
        raise ValueError('Sweep implementation differs from production baseline; do not change the lock')
    winner = max(rows, key=lambda r:r['macro_f05'])
    gain = winner['macro_f05'] - locked['macro_f05']
    result = {'rows':rows, 'winner':winner, 'locked':locked, 'gain':gain,
              'within_0_005':gain<=.005, 'within_0_01':gain<=.01,
              'sample_sha256':sample_hash,'pair_sha256':pair_hash,
              'floor_definitions':definitions,'fixed_params':params,
              'model_scoring_performed':False,'deadline_utc':args.deadline_utc}
    (out / 'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('WINNER',json.dumps(winner), 'GAIN',gain,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores', required=True)
    parser.add_argument('--floors', required=True)
    parser.add_argument('--deadline-utc', required=True)
    parser.add_argument('--output', default=str(Path(ARTIFACTS_DIR)/'decision_reverification'))
    run(parser.parse_args())
