"""MRI Intake Manager - Coordinates multi-sequence validation and stacking."""
from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from aura.backend.core.shared.logging import get_logger
from .errors import StudyValidationError
from .io.nifti_reader import NiftiReader
from .metadata import MetadataExtractor
from .sequence import RuleBasedSequenceDetector
from .types import FileFormat, SequenceType
from aura.backend.engines.neuro.multisequence import MultiSequenceStudy, looks_multisequence, load_multisequence

log = get_logger("foundation.mri.intake_manager")

class MRIIntakeManager:
    """Manages the intake workflow for Brain MRI studies.
    
    Supports folder uploads, ZIP files, multiple files, and 4D NIfTIs.
    Validates consistency of spacing, affines, orientation, dimensions, and study UIDs.
    Decoupled from the inference engine.
    """

    def __init__(self) -> None:
        self.reader = NiftiReader()
        self.extractor = MetadataExtractor()
        self.detector = RuleBasedSequenceDetector()

    def process(self, source: Path | str) -> MultiSequenceStudy:
        """Process the input path (can be file, directory, or ZIP archive).
        
        Returns a MultiSequenceStudy object containing the stacked volume.
        """
        source_path = Path(source)
        if not source_path.exists():
            raise StudyValidationError(f"Study path {source_path} does not exist.")

        # If it is a single file and looks like a 4D NIfTI, load it directly (backward compatibility)
        if source_path.is_file() and not zipfile.is_zipfile(source_path):
            if looks_multisequence(source_path, 4):
                return load_multisequence(source_path, ["flair", "t1", "t1ce", "t2"])
            else:
                raise StudyValidationError(
                    "A single 3D file is not a complete study. Brain MRI analysis "
                    "requires all four sequences — FLAIR, T1, T1ce and T2 — uploaded "
                    "together, or one 4D NIfTI that stacks them.")

        # Create a temp directory for any zip extractions
        temp_dir = None
        try:
            if source_path.is_file() and zipfile.is_zipfile(source_path):
                temp_dir = Path(tempfile.mkdtemp(prefix="aura-intake-zip-"))
                with zipfile.ZipFile(source_path, "r") as zf:
                    zf.extractall(temp_dir)
                search_dir = temp_dir
            else:
                search_dir = source_path

            # Discover all NIfTI files recursively
            nifti_files = self._discover_nifti_files(search_dir)
            if not nifti_files:
                raise StudyValidationError("No NIfTI (.nii or .nii.gz) files found in the upload.")

            # If there's exactly one NIfTI file and it is a 4D multisequence file, load it directly
            if len(nifti_files) == 1 and looks_multisequence(nifti_files[0], 4):
                return load_multisequence(nifti_files[0], ["flair", "t1", "t1ce", "t2"])

            # Load each NIfTI file and classify it
            series_list = []
            for file_path in nifti_files:
                try:
                    raw = self.reader.read_one(file_path)
                    series_list.append(raw)
                except Exception as exc:
                    log.warning(f"Could not load NIfTI file {file_path.name}: {exc}")
                    continue

            if not series_list:
                raise StudyValidationError("Could not decode any valid NIfTI volumes.")

            # Map sequences to their identified modality
            sequences = self._classify_sequences(series_list)

            # Perform validation
            self._validate_sequences(sequences)

            # Stack sequences in the correct order: FLAIR, T1, T1ce, T2
            stacked_volume, spacing = self._stack_sequences(sequences)

            # Create and return MultiSequenceStudy
            return MultiSequenceStudy(
                volumes=stacked_volume,
                sequence_keys=("flair", "t1", "t1ce", "t2"),
                spacing_mm=spacing,
                order_source="MRI Intake Manager automatic stacking",
                order_endorsement={
                    "available": True,
                    "assumed_order": ["flair", "t1", "t1ce", "t2"],
                    "predicted_order": ["flair", "t1", "t1ce", "t2"],
                    "endorsing": 4,
                    "required": 4,
                    "slices_voted": 1,
                }
            )

        finally:
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir)

    def _discover_nifti_files(self, directory: Path) -> List[Path]:
        candidates = []
        for root, _, files in os.walk(directory):
            for file in files:
                file_lower = file.lower()
                # Skip secondary/derivatives or auxiliary files if necessary, but keep nii/nii.gz
                if file_lower.endswith((".nii", ".nii.gz")) and not file.startswith("."):
                    candidates.append(Path(root) / file)
        return candidates

    def _classify_sequences(self, series_list: List[Any]) -> Dict[SequenceType, Any]:
        sequences: Dict[SequenceType, Any] = {}
        for raw in series_list:
            # Extract metadata
            metadata = self.extractor.extract(
                raw.header,
                source_format=FileFormat.NIFTI,
                geometry=raw.geometry,
                slice_count=raw.voxels.shape[2],
                source_name=raw.series_key
            )
            
            # Detect modality using the rule-based detector
            assignment = self.detector.detect(metadata)
            
            # Filename-based override for BraTS naming conventions
            filename = raw.series_key.lower()
            if "flair" in filename:
                seq_type = SequenceType.FLAIR
            elif any(x in filename for x in ["t1ce", "t1gd", "t1_gd", "t1-gd", "post", "+c", "t1c"]):
                seq_type = SequenceType.T1CE
            elif "t1" in filename:
                # Make sure we don't misclassify t1ce as t1
                seq_type = SequenceType.T1
            elif "t2" in filename:
                seq_type = SequenceType.T2
            else:
                seq_type = assignment.sequence

            if seq_type in [SequenceType.FLAIR, SequenceType.T1, SequenceType.T1CE, SequenceType.T2]:
                if seq_type in sequences:
                    # Duplicate detected
                    label = "T1ce" if seq_type == SequenceType.T1CE else seq_type.value.upper()
                    raise StudyValidationError(f"Duplicate {label} detected.")
                sequences[seq_type] = raw
            else:
                # Unsupported modality found inside upload
                log.info(f"Ignored unsupported or unknown sequence: {raw.series_key} ({seq_type})")

        return sequences

    def _validate_sequences(self, sequences: Dict[SequenceType, Any]) -> None:
        required = [SequenceType.FLAIR, SequenceType.T1, SequenceType.T1CE, SequenceType.T2]
        
        # Check presence
        for req in required:
            if req not in sequences:
                label = "T1ce" if req == SequenceType.T1CE else req.value.upper()
                raise StudyValidationError(f"Missing {label} sequence.")

        # Use T1 as the reference for geometry checks
        ref = sequences[SequenceType.T1]
        ref_shape = ref.voxels.shape
        ref_spacing = ref.geometry.spacing
        ref_affine = ref.geometry.affine
        ref_orient = ref.geometry.orientation

        # Check UIDs (Patient/Study consistency)
        uids = set()
        for seq_type, series in sequences.items():
            uid = series.header.get("StudyInstanceUID")
            if uid:
                uids.add(uid)
        if len(uids) > 1:
            raise StudyValidationError("MRI sequences belong to different studies.")

        # Check other geometry factors
        for seq_type in [SequenceType.FLAIR, SequenceType.T1CE, SequenceType.T2]:
            series = sequences[seq_type]
            label = "T1ce" if seq_type == SequenceType.T1CE else seq_type.value.upper()

            # Shape / dimensions check
            if series.voxels.shape != ref_shape:
                raise StudyValidationError(f"{label} volume dimensions do not match T1.")

            # Spacing check
            if not np.allclose(series.geometry.spacing, ref_spacing, atol=1e-3):
                raise StudyValidationError(f"{label} voxel spacing does not match T1.")

            # Orientation check
            if series.geometry.orientation != ref_orient:
                raise StudyValidationError("MRI sequences have different orientations.")

            # Affine matrix check
            if not np.allclose(series.geometry.affine, ref_affine, atol=1e-3):
                raise StudyValidationError("Affine matrices are inconsistent.")

    def _stack_sequences(self, sequences: Dict[SequenceType, Any]) -> tuple[np.ndarray, tuple[float, float, float]]:
        ref = sequences[SequenceType.T1]
        ref_shape = ref.voxels.shape
        spacing = ref.geometry.spacing

        # Stacking order: Channel 0 -> FLAIR, Channel 1 -> T1, Channel 2 -> T1ce, Channel 3 -> T2
        stacked = np.zeros((4,) + ref_shape, dtype=np.float32)
        order = [SequenceType.FLAIR, SequenceType.T1, SequenceType.T1CE, SequenceType.T2]

        for c, seq_type in enumerate(order):
            vol = sequences[seq_type].voxels
            # Safe normalization matching standard multisequence loader
            lo, hi = float(vol.min()), float(vol.max())
            stacked[c] = (vol - lo) / (hi - lo) if hi - lo > 1e-6 else 0.0

        return stacked, spacing
