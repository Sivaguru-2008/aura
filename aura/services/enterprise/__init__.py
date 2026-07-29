"""Hospital interoperability adapters — DICOM C-STORE intake, HL7 v2 and FHIR export.

Deliberately import-free: each adapter pulls in an optional integration dependency
(pynetdicom, hl7apy, fhir.resources), and a deployment that only needs FHIR should
not fail to start because the DICOM stack is absent. Import the submodule you want::

    from aura.services.enterprise.fhir import to_diagnostic_report
"""
