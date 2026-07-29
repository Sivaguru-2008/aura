from __future__ import annotations
import os
import tempfile
import threading
from pathlib import Path
from pynetdicom import AE, evt
from pynetdicom.sop_class import (
    ChestXRayImageStorage,
    MRImageStorage,
    ComputedRadiographyImageStorage,
    DigitalXRayImageStorageForPresentation,
)
import pydicom

class DicomListener:
    def __init__(self, dispatch_service, store, port=11112):
        self.dispatch = dispatch_service
        self.store = store
        self.port = port
        self.ae = AE(ae_title=b"AURA_PACS")
        
        # Add supported presentation contexts for storage SCP
        self.ae.add_supported_context(ChestXRayImageStorage)
        self.ae.add_supported_context(MRImageStorage)
        self.ae.add_supported_context(ComputedRadiographyImageStorage)
        self.ae.add_supported_context(DigitalXRayImageStorageForPresentation)
        
        self.server = None
        self._thread = None
        
    def start(self) -> None:
        def _run():
            handlers = [(evt.EVT_C_STORE, self.handle_c_store)]
            try:
                self.server = self.ae.start_server(("", self.port), block=False, evt_handlers=handlers)
                print(f"[DICOM Listener] Started C-STORE SCP on port {self.port}")
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

    def handle_c_store(self, event) -> int:
        """Handle a C-STORE request."""
        try:
            ds = event.dataset
            patient_id = getattr(ds, "PatientID", "unknown")
            
            # Save dataset to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp:
                pydicom.dcmwrite(tmp.name, ds)
                tmp_path = Path(tmp.name)
                
            # Process study in background thread to avoid blocking the DICOM association
            t = threading.Thread(
                target=self._process_dicom,
                args=(tmp_path, patient_id),
                daemon=True
            )
            t.start()
            return 0x0000 # Success status
        except Exception as e:
            print(f"[DICOM Listener] C-STORE exception: {e}")
            return 0xC000 # Processing Failure status

    def _process_dicom(self, path: Path, patient_id: str) -> None:
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
                    envelope = loop.run_until_complete(self.dispatch.dispatch(asset))
                    print(f"[DICOM Listener] Successfully routed/analyzed study {envelope.routing.study_id} (Modality: {envelope.routing.modality})")
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
