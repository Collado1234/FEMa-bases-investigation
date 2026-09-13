#!/usr/bin/env python3
"""Uso: python3 run_experiment.py configs/fema.yaml"""
import sys

from src.pipeline import run_experiment

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python3 run_experiment.py <config.yaml>")
        sys.exit(1)
    run_experiment(sys.argv[1])
