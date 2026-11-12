"""Bateria de testes estatisticos para a comparacao de funcoes de base do FEMa.

Camadas (ver METODOLOGIA.md):
  A. por dataset  (per_dataset.py)  - omnibus + pareada corrigida + equivalencia (TOST) + EDF (KS/CvM/AD)
  B. por grupo    (by_group.py)     - Friedman por grupo, dependencia ranking x contexto (permutacao), familias
  C. holdout      (delong.py)       - DeLong sobre AUC no teste (precisa de predicoes por amostra)
"""
