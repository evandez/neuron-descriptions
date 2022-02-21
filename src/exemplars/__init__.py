"""Tools for computing top-activating images in vision models.

The output of this process is `k` images for each unit, corresponding to
the `k` samples in a dataset that caused the strongest activation of the unit.
The activations can be visualized on the images with masks, and these in turn
are what are annotated by crowdworkers. See NetDissect: Quantifying
Interpretability of Deep Visual Representations [Bau et al., 2017] for a full
description of the method.

The models to compute exemplars for are configured in `models.py`, and the
datasets containing candidate exemplars are configured in `datasets.py`.
"""
# flake8: noqa
from src.exemplars import datasets, models
from src.exemplars.compute import discriminative, generative
