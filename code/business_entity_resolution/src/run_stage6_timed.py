"""Run full inference and its validator with an externally enforced deadline."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deadline-utc', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--candidate-input', required=True)
    args = parser.parse_args()
    deadline = dt.datetime.fromisoformat(args.deadline_utc).timestamp()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    status = {'deadline_utc': args.deadline_utc, 'status': 'running',
              'canonical_outputs_replaced': False}
    status_path = output / 'run_status.json'
    status_path.write_text(json.dumps(status, indent=2))
    command = [sys.executable, '-u', '-B', str(Path(__file__).with_name('run_stage6_inference.py')),
               '--output-dir', str(output), '--candidate-input', args.candidate_input]
    with (output / 'inference.log').open('w') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            code = proc.wait(timeout=max(0, deadline - time.time()))
            status.update(status='validated' if code == 0 else 'failed', exit_code=code)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            status.update(status='timed_out', exit_code=proc.returncode,
                          note='Partial files are not submission-ready; validation not completed.')
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
    status['finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
    status_path.write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
