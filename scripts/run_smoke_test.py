#!/usr/bin/env python
"""
Smoke test runner for RetinaX.
Runs all Phase 1-6 commands in order on configs/smoke.yaml.
Exits 0 only if all pass.
"""
import subprocess
import sys
import time
from pathlib import Path


def run_cmd(cmd, desc, timeout=300):
    """Run a command and return (success, output)."""
    print(f"\n{'='*60}")
    print(f"STEP: {desc}")
    print(f"CMD: {cmd}")
    print(f"{'='*60}")
    
    # Longer timeout for training steps
    if "train_quality" in cmd:
        timeout = 600  # 10 minutes for quality training
    elif "train_dr" in cmd:
        timeout = 300  # 5 minutes for DR training
    
    print(f"TIMEOUT: {timeout}s")
    
    start = time.time()
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        elapsed = time.time() - start
        if result.returncode == 0:
            print(f"PASS ({elapsed:.1f}s)")
            return True, result.stdout
        else:
            print(f"FAIL ({elapsed:.1f}s)")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            return False, result.stderr
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT ({timeout}s)")
        return False, "timeout"
    except Exception as e:
        print(f"ERROR: {e}")
        return False, str(e)


def main():
    config = "configs/smoke.yaml"
    steps = [
        # Phase 0: Scaffold (already done)
        # Phase 1: Data layer
        (f"python -m retinax.data.build_cache --config {config} --dataset aptos", "Build APTOS cache"),
        (f"python -m retinax.data.build_cache --config {config} --dataset idrid", "Build IDRiD cache"),
        (f"python -m retinax.data.build_cache --config {config} --dataset quality", "Build Quality cache"),
        (f"python -m retinax.data.build_splits --config {config}", "Build APTOS splits"),
        
        # Phase 2: Training
        (f"python -m retinax.train.train_dr --fold 0 --config {config}", "Train DR (fold 0)"),
        (f"python -m retinax.train.train_quality --config {config}", "Train Quality"),
        
        # Phase 3: Evaluation, calibration, uncertainty
        (f"python -m retinax.uncertainty.calibrate --config {config}", "Calibrate temperature"),
        (f"python -m retinax.uncertainty.analyze --config {config}", "Uncertainty analysis"),
        
        # Phase 5: Pipeline
        (f"python -m retinax.pipeline --config {config} --image smoke_data/DR_grading/train/20170331093312956.jpg", "Pipeline test"),
    ]
    
    print("RetinaX Smoke Test")
    print(f"Config: {config}")
    
    results = []
    for cmd, desc in steps:
        success, output = run_cmd(cmd, desc)
        results.append((desc, success, output))
        if not success:
            print(f"\n*** STOPPED AT: {desc} ***")
            break
    
    # Summary
    print(f"\n{'='*60}")
    print("SMOKE TEST SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for desc, success, _ in results:
        status = "PASS" if success else "FAIL"
        print(f"  {status}: {desc}")
        if not success:
            all_pass = False
    
    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()