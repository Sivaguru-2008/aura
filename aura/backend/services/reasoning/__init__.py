"""Longitudinal reasoning services — prior-study tracking and disease progression.

These sit above the per-study engines: they compare a new result against what AURA
has already recorded for the same patient, so they are only meaningful once the
study store holds more than one timepoint.
"""
