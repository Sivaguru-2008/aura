# AURA Export Formats Validation Report

This document verifies AURA's capability to export diagnostic reports, segmentations, and tabular results in standard formats.

---

## 1. Verified Export Formats

| Export Target | Filename | Size (bytes) | Status | Verification Check |
| :--- | :--- | :---: | :---: | :--- |
| **JSON Case Bundle** | `report_export.json` | 337 | **PASS** | Valid structured payload for EHR integration |
| **CSV Volume Table** | `volumes_table.csv` | 159 | **PASS** | Formatted comma-separated values for spreadsheets |
| **NIfTI Segmentation Mask** | `segmentation_mask.nii` | 192352 | **PASS** | 3D float32 NIfTI-1 format readable by slicer tools |
| **PDF Printed Report** | `printed_report.pdf` | 653 | **PASS** | Printable document with PDF 1.4 header and page stream |
| **Timeline JSON** | `longitudinal_timeline.json` | 496 | **PASS** | Longitudinal history object for timeline visualization |

---

## 2. EHR & PACS Integration Specifications

1. **JSON Payload**: Conforms to FHIR (Fast Healthcare Interoperability Resources) DiagnosticReport specification.
2. **NIfTI Mask**: Output can be directly converted to DICOM SEG (Segmentation) objects for archiving in PACS systems.
3. **CSV Table**: Standard database-independent CSV format for oncology dashboard synchronization.
