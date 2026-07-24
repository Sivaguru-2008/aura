"""Format readers: bytes on disk to a voxel array plus an affine plus a raw header.

Each reader implements :class:`~backend.foundation.mri.io.base.StudyReader` and does
exactly three things: decode the pixels, construct the RAS+ affine, and report what it
could not do. Readers never normalise intensities, never reorient, and never repair —
those are standardisation stages with their own audit entries, and a reader that
silently fixed something would make the processing history a lie.

Three formats, and no third-party imaging dependency for two of them:

* :mod:`~backend.foundation.mri.io.dicom_reader` — pydicom (already a dependency of
  the modality router).
* :mod:`~backend.foundation.mri.io.nifti_reader` — NIfTI-1 and NIfTI-2, implemented
  directly against the published header layout. nibabel is not installed in this
  deployment, and a foundation layer that cannot read the format the entire
  neuroimaging world exchanges data in is not a foundation layer.
* :mod:`~backend.foundation.mri.io.nrrd_reader` — NRRD, likewise implemented against
  the format specification (it is a text header over a raw or gzipped payload).

Both hand-written readers are covered by write-then-read round-trip tests that assert
world coordinates survive, which is the property that actually matters.
"""
from backend.foundation.mri.io.base import (
    RawSeries,
    SeriesIntegrity,
    SliceIssue,
    StudyReader,
)

__all__ = ["RawSeries", "SeriesIntegrity", "SliceIssue", "StudyReader"]
