"""Inference-only cache creation from the existing full-pool feature matrix."""
import argparse
import hashlib
import json
import signal
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

from config import ARTIFACTS_DIR
from decision import calibrated_probabilities
from features import FEATURE_NAMES


def main(deadline):
    remaining = datetime.fromisoformat(deadline).timestamp() - time.time()
    if remaining <= 0:
        raise TimeoutError('Hard stop reached')
    def stop(signum, frame):
        raise TimeoutError('Hard stop reached')
    signal.signal(signal.SIGALRM, stop)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    base = Path(ARTIFACTS_DIR) / 'embedding_fix'
    report = json.loads((base / 'diagnostic_results.json').read_text())
    saved = np.load(base / 'real_features.npz')
    if str(saved['pair_sha256']) != report['pair_sha256'] or list(saved['feature_names']) != FEATURE_NAMES:
        raise ValueError('Cached features do not match the trusted diagnostic')
    model_path = Path(ARTIFACTS_DIR) / 'model.joblib'
    calibration_path = Path(ARTIFACTS_DIR) / 'calibrator.joblib'
    for path, key in ((model_path,'model_sha256'),(calibration_path,'calibrator_sha256')):
        if hashlib.sha256(path.read_bytes()).hexdigest() != report[key]:
            raise ValueError(f'Changed artifact: {path}')
    matrix = saved['X']
    if matrix.shape != (report['pair_count'],len(FEATURE_NAMES)):
        raise ValueError('Unexpected feature matrix shape')
    model = joblib.load(model_path)
    calibration = joblib.load(calibration_path)
    if calibration['type'] != 'isotonic':
        raise ValueError('Expected the already-trained isotonic calibrator')
    started = time.time()
    probabilities = calibrated_probabilities(model, calibration, matrix)
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError('Invalid calibrated output')
    output = base / 'calibrated_probabilities.npz'
    temporary = base / 'calibrated_probabilities.pending.npz'
    np.savez_compressed(temporary, probabilities=probabilities,
                        pair_sha256=report['pair_sha256'],
                        model_sha256=report['model_sha256'],
                        calibrator_sha256=report['calibrator_sha256'],
                        sample_sha256=report['sample_sha256'])
    temporary.replace(output)
    print(json.dumps({'path':str(output),'pairs':len(probabilities),
                      'inference_seconds':time.time()-started,
                      'retraining':False,'blocking':False,'feature_extraction':False}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--deadline-utc',required=True)
    main(parser.parse_args().deadline_utc)
