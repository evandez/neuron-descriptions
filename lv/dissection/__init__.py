"""Functions for dissecting vision models.

This module contains tools for dissecting arbitrary CNNs.
Dissection outputs `k` images for each unit in the CNN, corresponding to
the `k` samples in a dataset that caused the strongest activation of the unit.
The activations can be visualized on the images with masks, and these in turn
are what are annotated by crowdworkers. See NetDissect: Quantifying
Interpretability of Deep Visual Representations [Bau et al., 2017] for a full
description of the method.

In addition to defining useful adapters for `third_party/netdissect`, this
module provides wrappers for many well known vision models and datasets
so that they can be plugged directly into dissection.
"""
