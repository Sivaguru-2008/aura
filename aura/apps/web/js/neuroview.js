/* AURA NeuroView: MRI-only vtk.js volume viewer. */
window.NEUROVIEW = (() => {
  "use strict";

  const NOT_AVAILABLE = "Not Available";
  const state = { caseId: null, payload: null, vtkReady: null, vtkView: null, layers: new Map() };
  const $ = (id) => document.getElementById(id);

  async function load(caseId, bundle) {
    const panel = $("panel-neuroview");
    if (!panel) return;
    const isMRI = String(bundle?.study_id || "").startsWith("STU-MR")
      || bundle?.fusion?.backend === "brain-vision-presence-head";
    if (!isMRI) {
      panel.hidden = true;
      teardown();
      return;
    }
    panel.hidden = false;
    if (state.caseId === caseId && state.payload) return;
    state.caseId = caseId;
    state.payload = null;
    $("nv-status").textContent = "loading NeuroView";
    try {
      const payload = await FX.api(`/v1/cases/${caseId}/neuroview`);
      state.payload = payload;
      render(payload);
    } catch {
      unavailable("Unable to compute from current MRI study.");
    }
  }

  function render(payload) {
    if (!payload || payload.status !== "available" || !payload.volume) {
      unavailable(payload?.message || "Unable to compute from current MRI study.");
      return;
    }
    $("nv-status").textContent = `${payload.source?.sequence_label || "MRI"} volume`;
    $("nv-meta").textContent = [
      payload.metadata?.spacing_mm ? `spacing ${payload.metadata.spacing_mm.map((v) => Number(v).toFixed(3)).join(" x ")} mm` : "spacing n/a",
      `model ${payload.metadata?.model_version || NOT_AVAILABLE}`,
    ].join(" | ");
    state.layers = new Map((payload.layers || []).map((l) => [l.key, decodeLayer(l)]));
    buildLayerControls(payload.layers || []);
    buildSliceControls(payload);
    bindPrimaryControls(payload);
    renderSlices();
    renderVtk(payload).catch(() => {
      $("nv-status").textContent = "GPU renderer unavailable";
    });
  }

  function unavailable(message) {
    $("panel-neuroview").hidden = false;
    $("nv-status").textContent = NOT_AVAILABLE;
    $("nv-meta").textContent = message || "Unable to compute from current MRI study.";
    $("nv-layers").innerHTML = "";
    ["nv-axial", "nv-coronal", "nv-sagittal"].forEach((id) => clearCanvas($(id)));
    teardown();
  }

  function decodeVolume(payload) {
    const dims = payload.volume.dims.map(Number);
    const bytes = base64ToBytes(payload.volume.data);
    return { dims, data: new Uint16Array(bytes.buffer), intensity: payload.volume.intensity };
  }

  function decodeLayer(layer) {
    if (!layer.available || !layer.data) return { ...layer, dataArray: null };
    const bytes = base64ToBytes(layer.data);
    return { ...layer, dataArray: new Uint8Array(bytes.buffer), enabled: true };
  }

  function base64ToBytes(text) {
    const bin = atob(text || "");
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  function buildLayerControls(layers) {
    const wrap = $("nv-layers");
    wrap.innerHTML = "";
    layers.forEach((layer) => {
      const label = document.createElement("label");
      label.className = `nv-layer ${layer.available ? "" : "off"}`;
      label.innerHTML = `
        <input type="checkbox" ${layer.available ? "checked" : "disabled"}>
        <span class="nv-swatch" style="background:${layer.color || "transparent"}"></span>
        <span>${layer.label}</span>
        <em>${layer.available ? `${layer.voxel_count} voxels` : NOT_AVAILABLE}</em>`;
      const input = label.querySelector("input");
      input.addEventListener("change", () => {
        const current = state.layers.get(layer.key);
        if (current) current.enabled = input.checked;
        renderSlices();
      });
      wrap.appendChild(label);
    });
  }

  function buildSliceControls(payload) {
    const { dims } = decodeVolume(payload);
    setRange("nv-axial-range", 0, dims[2] - 1, Math.floor((dims[2] - 1) / 2));
    setRange("nv-coronal-range", 0, dims[1] - 1, Math.floor((dims[1] - 1) / 2));
    setRange("nv-sagittal-range", 0, dims[0] - 1, Math.floor((dims[0] - 1) / 2));
    ["nv-axial-range", "nv-coronal-range", "nv-sagittal-range"].forEach((id) => {
      $(id).addEventListener("input", renderSlices);
    });
  }

  function bindPrimaryControls(payload) {
    const volOpacity = $("nv-opacity");
    const layerOpacity = $("nv-layer-opacity");
    const range = payload.volume.intensity;
    setRange("nv-window", 1, 65535, 65535);
    setRange("nv-level", 0, 65535, 32768);
    [volOpacity, layerOpacity, $("nv-window"), $("nv-level")].forEach((el) => {
      el.oninput = () => {
        renderSlices();
        updateVtkTransfer();
      };
    });
    $("nv-reset").onclick = () => {
      if (state.vtkView) {
        state.vtkView.renderer.resetCamera();
        state.vtkView.renderWindow.render();
      }
    };
    $("nv-intensity").textContent = range
      ? `window ${Number(range.window).toFixed(3)} | level ${Number(range.level).toFixed(3)}`
      : "window/level n/a";
  }

  function setRange(id, min, max, value) {
    const el = $(id);
    el.min = String(min);
    el.max = String(Math.max(min, max));
    el.value = String(Math.max(min, Math.min(max, value)));
  }

  async function renderVtk(payload) {
    const root = $("nv-vtk");
    root.innerHTML = "";
    const vtk = await loadVtk();
    const volume = decodeVolume(payload);
    const spacing = payload.metadata?.spacing_mm || [1, 1, 1];
    const fullScreenRenderer = vtk.Rendering.Misc.vtkFullScreenRenderWindow.newInstance({
      rootContainer: root,
      containerStyle: { height: "100%", width: "100%", position: "relative" },
      background: [0.02, 0.025, 0.04],
    });
    const imageData = vtk.Common.DataModel.vtkImageData.newInstance();
    imageData.setDimensions(...volume.dims);
    imageData.setSpacing(...spacing);
    imageData.getPointData().setScalars(vtk.Common.Core.vtkDataArray.newInstance({
      name: "MRI volume",
      values: volume.data,
      numberOfComponents: 1,
    }));
    const mapper = vtk.Rendering.Core.vtkVolumeMapper.newInstance();
    mapper.setInputData(imageData);
    const actor = vtk.Rendering.Core.vtkVolume.newInstance();
    actor.setMapper(mapper);
    const renderer = fullScreenRenderer.getRenderer();
    const renderWindow = fullScreenRenderer.getRenderWindow();
    renderer.addVolume(actor);
    state.vtkView = { vtk, actor, mapper, renderer, renderWindow, fullScreenRenderer };
    updateVtkTransfer();
    renderer.resetCamera();
    renderWindow.render();
  }

  function updateVtkTransfer() {
    const view = state.vtkView;
    if (!view) return;
    const vtk = view.vtk;
    const opacity = Number($("nv-opacity").value) / 100;
    const window = Number($("nv-window").value);
    const level = Number($("nv-level").value);
    const low = Math.max(0, level - window / 2);
    const high = Math.min(65535, level + window / 2);
    const ctfun = vtk.Rendering.Core.vtkColorTransferFunction.newInstance();
    ctfun.addRGBPoint(low, 0, 0, 0);
    ctfun.addRGBPoint(high, 1, 1, 1);
    const ofun = vtk.Common.DataModel.vtkPiecewiseFunction.newInstance();
    ofun.addPoint(low, 0.0);
    ofun.addPoint(high, opacity);
    view.actor.getProperty().setRGBTransferFunction(0, ctfun);
    view.actor.getProperty().setScalarOpacity(0, ofun);
    view.actor.getProperty().setInterpolationTypeToLinear();
    view.mapper.setSampleDistance(1.0);
    view.renderWindow.render();
  }

  async function loadVtk() {
    if (!state.vtkReady) {
      state.vtkReady = new Promise((resolve, reject) => {
        if (window.vtk) return resolve(window.vtk);
        const script = document.createElement("script");
        script.src = "https://unpkg.com/vtk.js";
        script.async = true;
        script.onload = () => window.vtk ? resolve(window.vtk) : reject(new Error("vtk.js unavailable"));
        script.onerror = reject;
        document.head.appendChild(script);
      });
    }
    return state.vtkReady;
  }

  function renderSlices() {
    if (!state.payload?.volume) return;
    const vol = decodeVolume(state.payload);
    drawPlane($("nv-axial"), vol, "axial", Number($("nv-axial-range").value));
    drawPlane($("nv-coronal"), vol, "coronal", Number($("nv-coronal-range").value));
    drawPlane($("nv-sagittal"), vol, "sagittal", Number($("nv-sagittal-range").value));
  }

  function drawPlane(canvas, vol, plane, index) {
    const [xDim, yDim, zDim] = vol.dims;
    const size = plane === "axial" ? [xDim, yDim] : plane === "coronal" ? [xDim, zDim] : [yDim, zDim];
    const [w, h] = size;
    const off = document.createElement("canvas");
    off.width = w; off.height = h;
    const ctx = off.getContext("2d");
    const img = ctx.createImageData(w, h);
    const window = Number($("nv-window").value);
    const level = Number($("nv-level").value);
    const low = Math.max(0, level - window / 2);
    const high = Math.min(65535, level + window / 2);
    const denom = Math.max(1, high - low);
    const layerAlpha = Number($("nv-layer-opacity").value) / 100;

    for (let row = 0; row < h; row++) {
      for (let col = 0; col < w; col++) {
        const voxel = plane === "axial"
          ? idx(col, row, index, xDim, yDim)
          : plane === "coronal"
            ? idx(col, index, row, xDim, yDim)
            : idx(index, col, row, xDim, yDim);
        const t = Math.max(0, Math.min(1, (vol.data[voxel] - low) / denom));
        let r = Math.round(t * 255), g = r, b = r;
        for (const layer of state.layers.values()) {
          if (!layer.enabled || !layer.dataArray || !layer.dataArray[voxel]) continue;
          const [lr, lg, lb] = hex(layer.color);
          r = Math.round(r * (1 - layerAlpha) + lr * layerAlpha);
          g = Math.round(g * (1 - layerAlpha) + lg * layerAlpha);
          b = Math.round(b * (1 - layerAlpha) + lb * layerAlpha);
        }
        const p = (row * w + col) * 4;
        img.data[p] = r; img.data[p + 1] = g; img.data[p + 2] = b; img.data[p + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    const dest = canvas.getContext("2d");
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(devicePixelRatio || 1, 2);
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    dest.setTransform(dpr, 0, 0, dpr, 0, 0);
    dest.imageSmoothingEnabled = false;
    dest.clearRect(0, 0, rect.width, rect.height);
    dest.drawImage(off, 0, 0, rect.width, rect.height);
  }

  function idx(x, y, z, xDim, yDim) {
    return x + y * xDim + z * xDim * yDim;
  }

  function hex(color) {
    const value = String(color || "#ffffff").replace("#", "");
    return [0, 2, 4].map((i) => parseInt(value.slice(i, i + 2), 16));
  }

  function clearCanvas(canvas) {
    const ctx = canvas?.getContext("2d");
    if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  function teardown() {
    if (state.vtkView?.fullScreenRenderer) {
      state.vtkView.fullScreenRenderer.delete();
    }
    state.vtkView = null;
    state.payload = null;
    const root = $("nv-vtk");
    if (root) root.innerHTML = "";
  }

  return { load };
})();
