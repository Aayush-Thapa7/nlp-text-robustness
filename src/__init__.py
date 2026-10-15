"""
nlp-text-robustness — src package
==================================

This package contains all reusable source code for the robustness experiments.
Keeping logic here (rather than inline in notebooks) makes the code testable,
importable, and easy to maintain.

Planned submodules
------------------
corruption.py
    Character-level text corruption utilities (deletion, insertion, swap,
    keyboard-proximity substitution, random substitution, repetition).

preprocessing.py
    Text cleaning, tokenization, and TF-IDF feature extraction helpers.

evaluation.py
    Functions to compute accuracy, F1, robustness-drop, and degradation curves.

visualization.py
    Plotting helpers for degradation curves, confusion matrices, and
    comparison bar charts.

Usage (from project root or a notebook)
-----------------------------------------
    from src.corruption import corrupt_text
    from src.evaluation import robustness_drop

Notes
-----
PyTorch / HuggingFace Transformers utilities will be added to this package
in Week 3, when transformer model evaluation begins.
"""

# Package version — increment as the project evolves
__version__ = "0.1.0"
__author__ = "nlp-text-robustness contributors"
