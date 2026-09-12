from __future__ import annotations
import fnmatch
import os
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Iterator
from pynetdicom import AE, evt
from pynetdicom.sop_class import (
    MRImageStorage,
    ComputedRadiographyImageStorage,
    DigitalXRayImageStorageForPresentation,
    PatientRootQueryRetrieveInformationModelFind,
    StudyRootQueryRetrieveInformationModelFind,
)
import pydicom

class DicomListener:
    def __init__(self, dispatch_service, store, port=11112):
        self.dispatch = dispatch_service
        self.store = store
        self.port = port
        self.ae = AE(ae_title=b"AURA_PACS")
        
        # Add supported presentation contexts for storage SCP
        self.ae.add_supported_context(MRImageStorage)
        self.ae.add_supported_context(ComputedRadiographyImageStorage)
        self.ae.add_supported_context(DigitalXRayImageStorageForPresentation)
        
        # Add supported presentation contexts for query find SCP
        self.ae.add_supported_context(PatientRootQueryRetrieveInformationModelFind)
        self.ae.add_supported_context(StudyRootQueryRetrieveInformationModelFind)
        
        self.server = None
        self._thread = None
        
    def start(self) -> None:
        def _run():
            handlers = [
                (evt.EVT_C_STORE, self.handle_c_store),
                (evt.EVT_C_FIND, self.handle_c_find),
            ]
            try:
                self.server = self.ae.start_server(("", self.port), block=False, evt_handlers=handlers)
                print(f"[DICOM Listener] Started C-STORE/C-FIND SCP on port {self.port}")
                self.server.serve_forever()
            except Exception as e:
                print(f"[DICOM Listener] Failed to start server: {e}")
            
        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        
    def stop(self) -> None:
        if self.server:
            try:
                self.server.shutdown()
                print("[DICOM Listener] C-STORE SCP stopped")
            except Exception as e:
                print(f"[DICOM Listener] Error during shutdown: {e}")

    @staticmethod
    def _resolve_tenant_id(event, ds: pydicom.dataset.Dataset | None = None) -> str:
        """Extract tenant identity from DICOM dataset tags or association AE titles."""
        if ds is not None:
            for tag_name in ("InstitutionName", "StationName", "InstitutionalDepartmentName"):
                val = getattr(ds, tag_name, None)
                if val:
                    cleaned = str(val).strip().lower().replace(" ", "_")
                    if cleaned and cleaned not in ("none", "unknown"):
                        return cleaned

        assoc = getattr(event, "assoc", None)
        if assoc:
            for entity in ("requestor", "acceptor"):
                actor = getattr(assoc, entity, None)
                if actor and hasattr(actor, "ae_title"):
                    aet = actor.ae_title
                    if isinstance(aet, bytes):
                        aet = aet.decode("ascii", errors="ignore")
                    aet = str(aet).strip()
                    if aet and aet.upper() not in ("ECHOSCU", "STORESCU", "FINDSCU", "AURA_PACS"):
                        return aet.lower()
        return "default"

    def handle_c_store(self, event) -> int:
        """Handle a C-STORE request."""
        try:
            ds = event.dataset
            patient_id = getattr(ds, "PatientID", "unknown")
            tenant_id = self._resolve_tenant_id(event, ds)
            
            # Save dataset to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp:
                pydicom.dcmwrite(tmp.name, ds)
                tmp_path = Path(tmp.name)
                
            # Process study in background thread to avoid blocking the DICOM association
            t = threading.Thread(
                target=self._process_dicom,
                args=(tmp_path, patient_id, tenant_id),
                daemon=True
            )
            t.start()
            return 0x0000 # Success status
        except Exception as e:
            print(f"[DICOM Listener] C-STORE exception: {e}")
            return 0xC000 # Processing Failure status

    def _process_dicom(self, path: Path, patient_id: str, tenant_id: str = "default") -> None:
        try:
            from aura.backend.core.upload.intake import stage_bytes
            payload = path.read_bytes()
            filename = f"dicom_upload_{patient_id}.dcm"
            
            with stage_bytes(payload, filename) as asset:
                import asyncio
                # Run the pipeline synchronously on the event loop for this background task
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    envelope = loop.run_until_complete(
                        self.dispatch.dispatch(asset, clinical_context={"tenant_id": tenant_id})
                    )
                    print(f"[DICOM Listener] Successfully routed/analyzed study {envelope.routing.study_id} (Modality: {envelope.routing.modality}) for tenant '{tenant_id}'")
                finally:
                    loop.close()
        except Exception as e:
            print(f"[DICOM Listener] Failed to process/dispatch received DICOM: {e}")
        finally:
            if path.exists():
                try:
                    os.remove(path)
                except Exception:
                    pass

    @staticmethod
    def _match(pattern: str, value: str) -> bool:
        """DICOM single-value matching: universal (empty) matches everything,
        otherwise wildcard matching per PS3.4 C.2.2.2.4 ('*' any run, '?' one char)."""
        if not pattern:
            return True
        return fnmatch.fnmatch((value or "").upper(), pattern.upper())

    @staticmethod
    def _study_datetime(created_at: str) -> tuple[str, str]:
        """Split a stored ISO timestamp into DICOM DA/TM. Empty when unparseable —
        a study date is a clinical fact, so an unknown one is returned zero-length
        rather than filled with today's date."""
        try:
            dt = datetime.fromisoformat(str(created_at))
        except (TypeError, ValueError):
            return "", ""
        return dt.strftime("%Y%m%d"), dt.strftime("%H%M%S")

    def handle_c_find(self, event) -> Iterator[tuple[int, pydicom.dataset.Dataset | None]]:
        """Handle a C-FIND query with tenant isolation."""
        try:
            ds = event.identifier
            qr_level = str(getattr(ds, "QueryRetrieveLevel", "") or "STUDY")
            tenant_id = self._resolve_tenant_id(event, ds)

            # Request keys. A key present-but-empty is a universal match and must
            # still be echoed back in the response; a key absent is not returned.
            requested = {elem.keyword for elem in ds if elem.keyword}
            q_patient_id = str(getattr(ds, "PatientID", "") or "").strip()
            q_study_uid = str(getattr(ds, "StudyInstanceUID", "") or "").strip()
            q_patient_name = str(getattr(ds, "PatientName", "") or "").strip()
            q_accession = str(getattr(ds, "AccessionNumber", "") or "").strip()

            matching_cases = self.store.list_cases(tenant_id=tenant_id)
            if not matching_cases and tenant_id == "default":
                matching_cases = self.store.list_cases()

            for c in matching_cases:
                if event.is_cancelled:
                    yield 0xFE00, None
                    return

                case_id = c["case_id"]
                study_id = c["study_id"]
                # Empty when the study arrived without a patient identifier.
                patient_id = str(c.get("patient_id", "") or "")

                if not self._match(q_patient_id, patient_id):
                    continue
                if not self._match(q_study_uid, study_id):
                    continue
                if not self._match(q_accession, case_id):
                    continue
                # AURA stores no patient name, so a name query can only be honoured
                # as a universal match. Echoing the queried name back as if it were
                # this record's name would fabricate a match.
                if q_patient_name:
                    continue

                study_date, study_time = self._study_datetime(c.get("created_at", ""))

                res = pydicom.dataset.Dataset()
                res.SpecificCharacterSet = "ISO_IR 192"
                res.QueryRetrieveLevel = qr_level
                known = {
                    "PatientID": patient_id,
                    "PatientName": "",           # not held by AURA
                    "PatientSex": "",            # not held by AURA
                    "PatientBirthDate": "",      # not held by AURA
                    "StudyInstanceUID": study_id,
                    "StudyID": study_id,
                    "AccessionNumber": case_id,
                    "StudyDate": study_date,
                    "StudyTime": study_time,
                    "ModalitiesInStudy": c.get("modality", "") or "",
                    "NumberOfStudyRelatedInstances": 1,
                }
                for keyword, value in known.items():
                    if keyword in requested:
                        setattr(res, keyword, value)

                yield 0xFF00, res

            yield 0x0000, None
        except Exception as e:
            print(f"[DICOM Listener] C-FIND exception: {e!r}")
            yield 0xC000, None
