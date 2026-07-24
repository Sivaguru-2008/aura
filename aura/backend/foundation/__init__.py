"""Foundation layers — modality-specific infrastructure shared by every engine.

A *foundation layer* is everything between raw uploaded bytes and a model-ready
study: reading, metadata, quality control, volume assembly, standardisation. It
contains no models and produces no clinical interpretation.

The separation exists because those two concerns have very different lifetimes. A
segmentation network is replaced every few months; the definition of "a brain MRI in
canonical RAS at 1 mm with a recorded processing history" is stable for years. Every
downstream NeuroMind module (segmentation, classification, registration, digital
twin, longitudinal analysis) consumes the *same* foundation output, so a
preprocessing fix lands in one place instead of in each model's data loader.

Currently one layer lives here: :mod:`backend.foundation.mri`.
"""
