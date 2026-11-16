/**
 * BATT-SIM // LG M50T SPMe DIGITAL TWIN RESEARCH WORKSTATION
 * 
 * =========================================================================
 * CENTRALIZED VISUAL CALIBRATION CONFIGURATION
 * =========================================================================
 * Tune all 3D mesh placement, particle containment boundaries, and particle
 * rendering parameters in this single configuration object.
 */
const VISUAL_CALIBRATION = {
  // 1. 3D Model Placement & Housing Transparency
  modelScale: 3.4,            // Uniform scale factor for the 45.glb cell in the viewport
  modelRotationY: 0.0,        // Y-axis rotation in radians (0.0 for canonical front elevation)
  modelOffsetY: -0.10,        // Vertical offset to seat model cleanly above testbed stage
  housingOpacityActive: 0.88, // Light, subtle translucency during Charge & Discharge only (0.85 - 0.90)
  housingOpacityIdle: 1.0,    // 100% solid/opaque in Idle mode (zero transparency in idle)

  // 2. Optical Brightness & PBR Surface Calibration
  // NOTE: Original 45.glb embedded material has metallicFactor: 0.80, which suppresses 80% of diffuse
  // light. Without an HDR environment map, high metalness reflects black void, making the cell
  // appear extremely dark. Setting modelMetalness to ~0.08 - 0.12 restores 90% diffuse reflection,
  // allowing the authentic texture colors (shaded.jpg) to be vividly illuminated by studio lights.
  modelMetalness: 0.10,       // Calibrated surface metalness (0.05 = soft dielectric, 0.15 = light sheen)
  modelRoughness: 0.48,       // Calibrated surface roughness (0.35 - 0.55)

  // 3. Laboratory Studio Lighting System (Трохи приглушене природне лабораторне освітлення)
  lighting: {
    ambientIntensity: 0.95,   // Omnidirectional base laboratory ambient light
    hemiIntensity: 1.05,      // Dual-hemisphere laboratory dome bounce (white ceiling / slate floor)
    keyLightIntensity: 1.40,  // Main frontal-top key directional light (casts clean soft shadow)
    fillLightIntensity: 0.95, // Front-left fill directional light to soften shadows
    rimLightIntensity: 0.80,  // Rear-top rim light outlining the cell contour against dark background
    toneMappingExposure: 1.22 // Camera exposure multiplier (1.10 - 1.35)
  },

  // 4. Li+ Particle Containment Box (Normalized Ratios from Bounding Box Min to Max)
  // [0.0 = bounding box minimum edge, 1.0 = bounding box maximum edge]
  ionBounds: {
    minXRatio: 0.08,          // Left margin (8% inset from Cathode exterior wall)
    maxXRatio: 0.92,          // Right margin (8% inset from Anode exterior wall)
    minYRatio: 0.06,          // Bottom margin (6% inset from bottom plate)
    maxYRatio: 0.62,          // Top boundary: capped at 62% of total height to strictly
                              // contain particles inside the lower rectangular cell body
                              // and prevent them from floating up into the top busbar/pipe
    minZRatio: 0.14,          // Front margin (14% inset from front face)
    maxZRatio: 0.86           // Back margin (14% inset from rear face)
  },

  // 5. Particle Density & Dynamics
  ionCount: 450,              // Total number of discrete Li+ point particles
  ionSize: 0.075,             // Render size of Li+ particles
  ionSpeedMultiplier: 1.0,    // Multiplier for longitudinal migration velocity

  // 6. Mode-Dependent Li+ Particle Point Colors (Strictly Point Particles Only)
  ionColors: {
    charge: '#38bdf8',        // Clean scientific cyan (Cathode -> Anode)
    discharge: '#f59e0b',     // Controlled warm amber (Anode -> Cathode)
    idle: '#94a3b8'           // Neutral laboratory slate (Brownian jitter in place)
  }
};

(function () {
  'use strict';

  // =========================================================================
  // 1. DATA ADAPTER LAYER (FastAPI Ready)
  // =========================================================================

  class SimulationDataProvider {
    constructor(config = {}) {
      this.baseUrl = config.baseUrl || 'http://localhost:8000/api';
      this.useLiveBackend = config.useLiveBackend || false;

      // LG M50T Cell Reference Constants (Chen et al., 2020)
      this.nominalCapacityAh = 5.0; // 5000 mAh
      this.upperCutoffV = 4.2;
      this.lowerCutoffV = 2.5;
      this.L_sei_init_nm = 5.0; // 5.0 nm initial SEI layer

      this.checkBackendHealth();
    }

    async checkBackendHealth() {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 1200);
        const res = await fetch(`${this.baseUrl}/health`, { signal: controller.signal });
        clearTimeout(timeoutId);
        if (res.ok) {
          this.useLiveBackend = true;
          this._updateBackendBadge(true);
          return true;
        }
      } catch (e) {
        this.useLiveBackend = false;
        this._updateBackendBadge(false);
      }
      return false;
    }

    _updateBackendBadge(isLive) {
      const badge = document.getElementById('status-backend-badge');
      const healthEp = document.getElementById('ep-health-status');
      if (badge) {
        if (isLive) {
          badge.textContent = 'CONNECTED // FASTAPI LIVE';
          badge.style.color = 'var(--accent-cyan)';
          if (healthEp) {
            healthEp.textContent = 'ONLINE (200 OK)';
            healthEp.style.color = 'var(--accent-emerald)';
          }
        } else {
          badge.textContent = 'FASTAPI READY (LOCAL SPMe)';
          badge.style.color = 'var(--accent-emerald)';
          if (healthEp) {
            healthEp.textContent = 'STANDBY';
            healthEp.style.color = 'var(--accent-cyan)';
          }
        }
      }
    }

    /**
     * Analytical OCV functions for LG M50T (Chen2020)
     * Exact equations from backend/physics/config.py
     */
    _ocv_n(theta) {
      return (
        1.9793 * Math.exp(-39.3631 * theta) +
        0.2482 -
        0.0909 * Math.tanh(29.8538 * (theta - 0.1234)) -
        0.04478 * Math.tanh(14.9159 * (theta - 0.2769)) -
        0.0205 * Math.tanh(30.4444 * (theta - 0.6103))
      );
    }

    _ocv_p(theta) {
      return (
        -0.8090 * theta +
        4.4875 -
        0.0428 * Math.tanh(18.5138 * (theta - 0.5542)) -
        17.7326 * Math.tanh(15.7890 * (theta - 0.3117)) +
        17.5842 * Math.tanh(15.9308 * (theta - 0.3120))
      );
    }

    /**
     * Fetch or compute step-by-step cycle transient telemetry.
     * Schema strictly matches BatterySimulation.diagnostics()
     */
    async fetchCycleStep(params) {
      if (this.useLiveBackend) {
        try {
          const res = await fetch(`${this.baseUrl}/simulate/step`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
          });
          if (res.ok) return await res.json();
        } catch (err) {
          console.warn('[DataProvider] Remote step call failed, using local model:', err);
        }
      }

      // Local Electrochemical Transient Approximation
      const { mode, current, soc, timeElapsed, delta } = params;
      const speedUp = 25.0; // Interactive acceleration
      const socFraction = Math.max(0.001, Math.min(0.999, soc / 100.0));

      const theta_n = 0.025 + 0.88 * socFraction;
      const theta_p = 0.915 - 0.64 * socFraction;
      const Ueq = this._ocv_p(theta_p) - this._ocv_n(theta_n);

      const R_ohm = 0.028; // Ohms
      let eta = 0.0;
      let newSoc = soc;
      let appliedCurrent = 0.0;

      if (mode === 'charge') {
        appliedCurrent = -current;
        const deltaSoc = (current / (this.nominalCapacityAh * 3600.0)) * 100.0 * delta * speedUp;
        newSoc = Math.min(100.0, soc + deltaSoc);
        eta = current * R_ohm + 0.015 * Math.log1p(current);
      } else if (mode === 'discharge') {
        appliedCurrent = current;
        const deltaSoc = (current / (this.nominalCapacityAh * 3600.0)) * 100.0 * delta * speedUp;
        newSoc = Math.max(0.0, soc - deltaSoc);
        eta = - (current * R_ohm + 0.015 * Math.log1p(current));
      }

      const voltage = Math.max(this.lowerCutoffV, Math.min(this.upperCutoffV, Ueq + eta));

      // Diffusion-limited SEI growth: L_sei = sqrt(L_0^2 + 2*k*t)
      const tTot = Math.max(0, timeElapsed);
      const k_sei = 1.6e-5; // nm^2/s effective rate
      const sei_thickness_nm = Math.sqrt(this.L_sei_init_nm * this.L_sei_init_nm + 2.0 * k_sei * tTot);
      const capacity_loss_ah = (sei_thickness_nm - this.L_sei_init_nm) * 0.018;

      return {
        time_s: timeElapsed,
        voltage_V: voltage,
        soc_percent: newSoc,
        sei_thickness_nm: sei_thickness_nm,
        capacity_loss_ah: capacity_loss_ah,
        current_A: appliedCurrent
      };
    }

    /**
     * Fetch or compute multi-cycle lifecycle degradation trajectory.
     * Schema strictly matches BatterySimulation.run_cycles()["cycle_summary"]
     */
    async fetchLifecycleSimulation(params) {
      if (this.useLiveBackend) {
        try {
          const res = await fetch(`${this.baseUrl}/simulate/lifecycle`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
          });
          if (res.ok) return await res.json();
        } catch (err) {
          console.warn('[DataProvider] Remote lifecycle call failed, using local model:', err);
        }
      }

      const nCycles = params.n_cycles || 100;
      const current = params.current || 5.0;

      const cycles = [];
      const capacities = [];
      const soh_percent = [];
      const capacity_fade_percent = [];
      const sei_thickness_nm = [];
      const capacity_loss_ah = [];

      const Q_init = 4.885;

      for (let c = 1; c <= nCycles; c++) {
        cycles.push(c);
        const sei_nm = this.L_sei_init_nm + 1.25 * Math.sqrt(c);
        const q_loss = 0.0085 * Math.sqrt(c);
        const q_dis = Math.max(3.2, Q_init - q_loss * (1.0 + 0.12 * (current / 5.0)));
        const soh = (q_dis / Q_init) * 100.0;
        const fade = 100.0 - soh;

        capacities.push(Number(q_dis.toFixed(4)));
        soh_percent.push(Number(soh.toFixed(3)));
        capacity_fade_percent.push(Number(fade.toFixed(3)));
        sei_thickness_nm.push(Number(sei_nm.toFixed(2)));
        capacity_loss_ah.push(Number(q_loss.toFixed(4)));
      }

      return {
        cycle: cycles,
        discharge_capacity_Ah: capacities,
        SOH_percent: soh_percent,
        capacity_fade_percent: capacity_fade_percent,
        sei_thickness_nm: sei_thickness_nm,
        capacity_loss_ah: capacity_loss_ah
      };
    }
  }

  // =========================================================================
  // 2. WORKSTATION STATE & CONTROLLERS
  // =========================================================================

  const state = {
    mode: 'idle', // 'idle' | 'charge' | 'discharge'
    isRunning: false,
    soc: 65.0,
    current: 5.0,
    voltage: 3.782,
    seiThicknessNm: 5.0,
    capacityLossAh: 0.0,
    soh: 100.0,
    timeElapsed: 0,
    autoRotate: false
  };

  const dataProvider = new SimulationDataProvider();

  // DOM Elements
  const container = document.getElementById('canvas-container');
  const engineStatusTag = document.getElementById('engine-status-tag');
  const engineStatusLabel = document.getElementById('engine-status-label');

  // HUD Readouts
  const hudModePill = document.getElementById('hud-mode-pill');
  const hudModeText = document.getElementById('hud-mode-text');
  const hudVoltage = document.getElementById('hud-voltage');
  const hudSoc = document.getElementById('hud-soc');
  const hudCurrent = document.getElementById('hud-current');
  const hudSei = document.getElementById('hud-sei');
  const hudQlli = document.getElementById('hud-qlli');
  const metricSohVal = document.getElementById('metric-soh-val');

  // Input & Buttons
  const btnModeIdle = document.getElementById('btn-mode-idle');
  const btnModeCharge = document.getElementById('btn-mode-charge');
  const btnModeDischarge = document.getElementById('btn-mode-discharge');
  const modeButtons = [btnModeIdle, btnModeCharge, btnModeDischarge];

  const sliderCurrent = document.getElementById('slider-current');
  const inputCurrent = document.getElementById('input-current');
  const readoutCurrent = document.getElementById('readout-current');

  const btnRun = document.getElementById('btn-run');
  const btnPause = document.getElementById('btn-pause');
  const btnReset = document.getElementById('btn-reset');
  const btnRunLifecycle = document.getElementById('btn-run-lifecycle');
  const btnCameraReset = document.getElementById('btn-camera-reset');
  const btnAutoRotate = document.getElementById('btn-autorotate-toggle');

  // =========================================================================
  // 3. THREE.JS 3D SCENE SETUP
  // =========================================================================

  let scene, camera, renderer, controls;
  let batteryGroup;
  const batteryMaterials = []; // References to mesh materials to dynamically switch transparency

  // Li+ Ion Transport System
  let ionParticleSystem;
  let ionPositions, ionInitialPositions, ionColors;
  let calculatedIonBounds = { minX: -1.3, maxX: 1.3, minY: -0.5, maxY: 0.2, minZ: -0.3, maxZ: 0.3 };
  const clock = new THREE.Clock();

  // 3D Spatial Anchors for Labels
  let spatialAnchors = [];

  function initThreeScene() {
    scene = new THREE.Scene();
    scene.background = null;

    const aspect = container.clientWidth / container.clientHeight;
    camera = new THREE.PerspectiveCamera(40, aspect, 0.1, 100);
    camera.position.set(0.0, 1.8, 5.0);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = (VISUAL_CALIBRATION.lighting && VISUAL_CALIBRATION.lighting.toneMappingExposure) || 1.35;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.minDistance = 2.0;
    controls.maxDistance = 11.0;
    controls.maxPolarAngle = Math.PI / 2 + 0.15;
    controls.target.set(0, 0, 0);

    // Clean neutral studio lighting
    setupLaboratoryLighting();

    // Measurement stage floor
    setupTestbedEnvironment();

    // Main battery group
    batteryGroup = new THREE.Group();
    scene.add(batteryGroup);

    // Load 3D Cell (assets/45.glb or procedural fallback)
    loadBatteryModel();

    window.addEventListener('resize', onWindowResize);
  }

  function setupLaboratoryLighting() {
    const l = VISUAL_CALIBRATION.lighting || {
      ambientIntensity: 0.95,
      hemiIntensity: 1.05,
      keyLightIntensity: 1.40,
      fillLightIntensity: 0.95,
      rimLightIntensity: 0.80
    };

    // 1. Ambient laboratory base illumination
    const ambientLight = new THREE.AmbientLight(0xffffff, l.ambientIntensity);
    scene.add(ambientLight);

    // 2. Dual-hemisphere dome lighting: simulates laboratory ceiling diffused light (top)
    // and ground slate bounce (bottom), preventing any angle from appearing pitch black
    const hemiLight = new THREE.HemisphereLight(0xffffff, 0x334155, l.hemiIntensity);
    hemiLight.position.set(0, 15, 0);
    scene.add(hemiLight);

    // 3. Primary Key Light (frontal-top-right)
    const keyLight = new THREE.DirectionalLight(0xffffff, l.keyLightIntensity);
    keyLight.position.set(5, 8, 6);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.width = 1024;
    keyLight.shadow.mapSize.height = 1024;
    keyLight.shadow.bias = -0.0005;
    scene.add(keyLight);

    // 4. Fill Light (frontal-left) to illuminate shadows
    const fillLight = new THREE.DirectionalLight(0xffffff, l.fillLightIntensity);
    fillLight.position.set(-5, 4, 4);
    scene.add(fillLight);

    // 5. Rim / Edge Backlight (rear-top) to crisply define battery silhouettes
    const rimLight = new THREE.DirectionalLight(0xffffff, l.rimLightIntensity);
    rimLight.position.set(0, 6, -5);
    scene.add(rimLight);
  }

  function setupTestbedEnvironment() {
    const diskGeo = new THREE.CircleGeometry(3.6, 64);
    const diskMat = new THREE.MeshStandardMaterial({
      color: 0x0f151e,
      roughness: 0.85,
      metalness: 0.2,
      transparent: true,
      opacity: 0.65
    });
    const disk = new THREE.Mesh(diskGeo, diskMat);
    disk.rotation.x = -Math.PI / 2;
    disk.position.y = -1.35;
    disk.receiveShadow = true;
    scene.add(disk);

    const ringGeo = new THREE.RingGeometry(1.2, 3.4, 64);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x242f3d,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.35,
      wireframe: true
    });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = -1.34;
    scene.add(ring);
  }

  // =========================================================================
  // 4. MODEL LOADING & DYNAMIC TRANSLUCENCY (ORIGINAL MATERIALS UNTOUCHED)
  // =========================================================================

  function updateHousingTransparency(mode) {
    const isActive = (mode === 'charge' || mode === 'discharge');
    const targetOpacity = isActive ? VISUAL_CALIBRATION.housingOpacityActive : VISUAL_CALIBRATION.housingOpacityIdle;
    const isTransparent = isActive && targetOpacity < 0.999;

    batteryMaterials.forEach((mat) => {
      mat.transparent = isTransparent;
      mat.opacity = targetOpacity;
      mat.depthWrite = !isTransparent; // depthWrite: false during active mode ensures internal particles show cleanly
      mat.needsUpdate = true;
    });
  }

  function loadBatteryModel() {
    const dracoLoader = new THREE.DRACOLoader();
    dracoLoader.setDecoderPath('https://www.gstatic.com/draco/versioned/decoders/1.4.1/');
    dracoLoader.setDecoderConfig({ type: 'js' });

    const gltfLoader = new THREE.GLTFLoader();
    gltfLoader.setDRACOLoader(dracoLoader);

    const candidates = ['assets/45.glb', 'assets/battery.glb'];

    function tryLoad(index) {
      if (index >= candidates.length) {
        console.warn('[Three.js] Using procedural 21700 cell fallback.');
        createProceduralBattery();
        return;
      }

      const path = candidates[index];
      gltfLoader.load(
        path,
        function (gltf) {
          console.log('[Three.js] 3D cell loaded successfully from:', path);
          processLoadedModel(gltf.scene);
        },
        undefined,
        function (err) {
          console.warn(`[Three.js] Candidate failed ${path}:`, err);
          tryLoad(index + 1);
        }
      );
    }

    tryLoad(0);
  }

  function processLoadedModel(modelScene) {
    batteryGroup.clear();
    batteryMaterials.length = 0;

    // 1. Wrapper group to center and apply VISUAL_CALIBRATION transform
    const wrapper = new THREE.Group();
    wrapper.add(modelScene);

    // Compute raw model bounding box to find its geometric centroid
    const initialBbox = new THREE.Box3().setFromObject(modelScene);
    const center = new THREE.Vector3();
    const size = new THREE.Vector3();
    initialBbox.getCenter(center);
    initialBbox.getSize(size);

    // Center model at local origin (0, 0, 0)
    modelScene.position.sub(center);

    // Apply scale, rotation, and offset from VISUAL_CALIBRATION
    const maxDim = Math.max(size.x, size.y, size.z);
    const targetScale = VISUAL_CALIBRATION.modelScale / (maxDim || 1);
    wrapper.scale.set(targetScale, targetScale, targetScale);
    wrapper.rotation.y = VISUAL_CALIBRATION.modelRotationY;
    wrapper.position.y = VISUAL_CALIBRATION.modelOffsetY;

    batteryGroup.add(wrapper);
    batteryGroup.updateMatrixWorld(true);

    // 2. Keep embedded materials from 45.glb completely untouched.
    // Configure opacity dynamically: light translucency (0.88) in Charge/Discharge, 100% solid in Idle.
    modelScene.traverse((node) => {
      if (node.isMesh && node.material) {
        node.castShadow = true;
        node.receiveShadow = true;

        // Clone material to preserve original texture and base colors
        node.material = node.material.clone();
        node.material.depthTest = true;

        // Optical Brightness Calibration:
        // Override high metallicFactor (0.80) which cancels 80% of diffuse light.
        // Calibrating metalness to ~0.10 restores 90% diffuse reflection, rendering
        // the authentic texture colors (shaded.jpg) brightly under studio lights.
        if (VISUAL_CALIBRATION.modelMetalness !== undefined) {
          node.material.metalness = VISUAL_CALIBRATION.modelMetalness;
        }
        if (VISUAL_CALIBRATION.modelRoughness !== undefined) {
          node.material.roughness = VISUAL_CALIBRATION.modelRoughness;
        }

        // Zero emissive glow: housing remains 100% static in its authentic color
        if (node.material.emissive) {
          node.material.emissive.set(0x000000);
          node.material.emissiveIntensity = 0.0;
        }

        batteryMaterials.push(node.material);
      }
    });

    updateHousingTransparency(state.mode);

    // 3. Compute final world-space bounding box of the cell
    const cellBbox = new THREE.Box3().setFromObject(batteryGroup);

    // 4. Initialize particle containment strictly within calibrated bounds
    setupLithiumIonSystemInBox(cellBbox);

    // 5. Initialize 3D Spatial Anchors for labels
    setupSpatialAnchors(cellBbox);
  }

  function createProceduralBattery() {
    batteryGroup.clear();
    batteryMaterials.length = 0;

    const cellGroup = new THREE.Group();
    const bodyRadius = 0.72;
    const bodyLength = 3.3;

    // Horizontal cylinder along X axis
    const bodyGeo = new THREE.CylinderGeometry(bodyRadius, bodyRadius, bodyLength, 64);
    bodyGeo.rotateZ(Math.PI / 2);

    const bodyMat = new THREE.MeshStandardMaterial({
      color: 0x94a3b8,
      metalness: 0.12,
      roughness: 0.35,
      depthTest: true
    });
    batteryMaterials.push(bodyMat);

    const bodyMesh = new THREE.Mesh(bodyGeo, bodyMat);
    cellGroup.add(bodyMesh);

    // Terminal Caps
    const terminalMat = new THREE.MeshStandardMaterial({
      color: 0xcfd8dc,
      metalness: 0.30,
      roughness: 0.25
    });

    const nubGeo = new THREE.CylinderGeometry(0.24, 0.24, 0.18, 32);
    nubGeo.rotateZ(Math.PI / 2);
    const nubMesh = new THREE.Mesh(nubGeo, terminalMat);
    nubMesh.position.x = - (bodyLength / 2 + 0.09);
    cellGroup.add(nubMesh);

    const flatCapGeo = new THREE.CylinderGeometry(bodyRadius * 0.96, bodyRadius * 0.96, 0.12, 64);
    flatCapGeo.rotateZ(Math.PI / 2);
    const flatCapMesh = new THREE.Mesh(flatCapGeo, terminalMat);
    flatCapMesh.position.x = bodyLength / 2 + 0.06;
    cellGroup.add(flatCapMesh);

    cellGroup.position.y = VISUAL_CALIBRATION.modelOffsetY;
    cellGroup.rotation.y = VISUAL_CALIBRATION.modelRotationY;
    batteryGroup.add(cellGroup);
    batteryGroup.updateMatrixWorld(true);

    updateHousingTransparency(state.mode);

    const cellBbox = new THREE.Box3().setFromObject(batteryGroup);
    setupLithiumIonSystemInBox(cellBbox);
    setupSpatialAnchors(cellBbox);
  }

  // =========================================================================
  // 5. CALIBRATED Li+ ION CONTAINMENT SYSTEM
  // =========================================================================

  function createCrispIonTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = 32;
    canvas.height = 32;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, 32, 32);

    // Crisp round disc without diffuse bloom halo
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(16, 16, 11, 0, Math.PI * 2);
    ctx.fill();

    return new THREE.CanvasTexture(canvas);
  }

  function setupLithiumIonSystemInBox(cellBbox) {
    if (ionParticleSystem) {
      scene.remove(ionParticleSystem);
      if (ionParticleSystem.geometry) ionParticleSystem.geometry.dispose();
      if (ionParticleSystem.material) ionParticleSystem.material.dispose();
    }

    const sizeX = cellBbox.max.x - cellBbox.min.x;
    const sizeY = cellBbox.max.y - cellBbox.min.y;
    const sizeZ = cellBbox.max.z - cellBbox.min.z;
    const b = VISUAL_CALIBRATION.ionBounds;

    // Derive particle boundaries using VISUAL_CALIBRATION ratios
    calculatedIonBounds = {
      minX: cellBbox.min.x + sizeX * b.minXRatio,
      maxX: cellBbox.min.x + sizeX * b.maxXRatio,
      minY: cellBbox.min.y + sizeY * b.minYRatio,
      maxY: cellBbox.min.y + sizeY * b.maxYRatio, // excludes the top busbar/pipe connector
      minZ: cellBbox.min.z + sizeZ * b.minZRatio,
      maxZ: cellBbox.min.z + sizeZ * b.maxZRatio
    };

    const spanX = calculatedIonBounds.maxX - calculatedIonBounds.minX;
    const spanY = calculatedIonBounds.maxY - calculatedIonBounds.minY;
    const spanZ = calculatedIonBounds.maxZ - calculatedIonBounds.minZ;

    const count = VISUAL_CALIBRATION.ionCount;
    const geo = new THREE.BufferGeometry();
    ionPositions = new Float32Array(count * 3);
    ionInitialPositions = new Float32Array(count * 3);
    ionColors = new Float32Array(count * 3);

    const initialColor = new THREE.Color(VISUAL_CALIBRATION.ionColors.idle);

    for (let i = 0; i < count; i++) {
      const idx = i * 3;
      const x = calculatedIonBounds.minX + Math.random() * spanX;
      const y = calculatedIonBounds.minY + Math.random() * spanY;
      const z = calculatedIonBounds.minZ + Math.random() * spanZ;

      ionPositions[idx] = x;
      ionPositions[idx + 1] = y;
      ionPositions[idx + 2] = z;

      ionInitialPositions[idx] = x;
      ionInitialPositions[idx + 1] = y;
      ionInitialPositions[idx + 2] = z;

      ionColors[idx] = initialColor.r;
      ionColors[idx + 1] = initialColor.g;
      ionColors[idx + 2] = initialColor.b;
    }

    geo.setAttribute('position', new THREE.BufferAttribute(ionPositions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(ionColors, 3));

    const particleTexture = createCrispIonTexture();
    const mat = new THREE.PointsMaterial({
      size: VISUAL_CALIBRATION.ionSize,
      map: particleTexture,
      vertexColors: true,
      transparent: true,
      opacity: 0.95,
      depthWrite: false,
      depthTest: true
    });

    ionParticleSystem = new THREE.Points(geo, mat);
    scene.add(ionParticleSystem);
  }

  function updateLithiumIons(delta) {
    if (!ionParticleSystem || !ionPositions) return;

    const positions = ionParticleSystem.geometry.attributes.position.array;
    const colors = ionParticleSystem.geometry.attributes.color.array;
    const time = clock.getElapsedTime();
    const count = VISUAL_CALIBRATION.ionCount;

    const spanX = calculatedIonBounds.maxX - calculatedIonBounds.minX;

    // Migration along longitudinal X axis:
    // Charge: Cathode (-X) -> Anode (+X) => direction = +1
    // Discharge: Anode (+X) -> Cathode (-X) => direction = -1
    // Idle: direction = 0 (subtle thermal jitter in place)
    let direction = 0;
    let baseSpeed = 0.0;

    if (state.isRunning) {
      if (state.mode === 'charge') {
        direction = 1;
        baseSpeed = (0.35 + (state.current / 10.0) * 1.4) * VISUAL_CALIBRATION.ionSpeedMultiplier;
      } else if (state.mode === 'discharge') {
        direction = -1;
        baseSpeed = (0.35 + (state.current / 10.0) * 1.4) * VISUAL_CALIBRATION.ionSpeedMultiplier;
      }
    }

    // Set particle color from VISUAL_CALIBRATION
    const targetColor = new THREE.Color();
    if (state.mode === 'charge') {
      targetColor.set(VISUAL_CALIBRATION.ionColors.charge);
    } else if (state.mode === 'discharge') {
      targetColor.set(VISUAL_CALIBRATION.ionColors.discharge);
    } else {
      targetColor.set(VISUAL_CALIBRATION.ionColors.idle);
    }

    for (let i = 0; i < count; i++) {
      const idx = i * 3;

      if (direction !== 0) {
        positions[idx] += direction * baseSpeed * delta;
        if (positions[idx] > calculatedIonBounds.maxX) positions[idx] -= spanX;
        if (positions[idx] < calculatedIonBounds.minX) positions[idx] += spanX;
      } else {
        positions[idx] = ionInitialPositions[idx] + Math.sin(time * 2.5 + i) * 0.015;
      }

      // Strict containment clamp: particles never poke out of the battery body
      positions[idx] = Math.max(calculatedIonBounds.minX, Math.min(calculatedIonBounds.maxX, positions[idx]));

      const curY = ionInitialPositions[idx + 1] + Math.sin(time * 2.8 + i * 0.4) * 0.012;
      const curZ = ionInitialPositions[idx + 2] + Math.cos(time * 2.6 + i * 0.5) * 0.012;
      positions[idx + 1] = Math.max(calculatedIonBounds.minY, Math.min(calculatedIonBounds.maxY, curY));
      positions[idx + 2] = Math.max(calculatedIonBounds.minZ, Math.min(calculatedIonBounds.maxZ, curZ));

      colors[idx] = targetColor.r;
      colors[idx + 1] = targetColor.g;
      colors[idx + 2] = targetColor.b;
    }

    ionParticleSystem.geometry.attributes.position.needsUpdate = true;
    ionParticleSystem.geometry.attributes.color.needsUpdate = true;
  }

  // =========================================================================
  // 6. 3D-ANCHORED SCREEN-PROJECTED STRUCTURAL LABELS
  // =========================================================================

  function setupSpatialAnchors(cellBbox) {
    spatialAnchors = [];

    // Derive 3D positions directly from cell bounding box
    const xCathode = cellBbox.min.x + (cellBbox.max.x - cellBbox.min.x) * 0.22;
    const xSeparator = (cellBbox.min.x + cellBbox.max.x) * 0.5;
    const xAnode = cellBbox.min.x + (cellBbox.max.x - cellBbox.min.x) * 0.78;

    const yTop = cellBbox.max.y;
    const zMid = (cellBbox.min.z + cellBbox.max.z) * 0.5;

    const configs = [
      {
        id: 'badge-cathode',
        lineId: 'line-cathode',
        dotId: 'dot-cathode',
        color: '#38bdf8',
        targetPos: new THREE.Vector3(xCathode, yTop, zMid),
        anchorPos: new THREE.Vector3(xCathode, yTop + 0.52, zMid)
      },
      {
        id: 'badge-separator',
        lineId: 'line-separator',
        dotId: 'dot-separator',
        color: '#94a3b8',
        targetPos: new THREE.Vector3(xSeparator, yTop, zMid),
        anchorPos: new THREE.Vector3(xSeparator, yTop + 0.72, zMid) // elevated to prevent badge overlap
      },
      {
        id: 'badge-anode',
        lineId: 'line-anode',
        dotId: 'dot-anode',
        color: '#f59e0b',
        targetPos: new THREE.Vector3(xAnode, yTop, zMid),
        anchorPos: new THREE.Vector3(xAnode, yTop + 0.52, zMid)
      }
    ];

    // Build SVG leader lines and target dots in overlay
    const svgOverlay = document.getElementById('labels-svg-overlay');
    if (svgOverlay) {
      svgOverlay.innerHTML = '';
      configs.forEach(c => {
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('id', c.lineId);
        line.setAttribute('stroke', c.color);
        line.setAttribute('stroke-width', '1');
        line.setAttribute('stroke-dasharray', '2,2');
        line.setAttribute('opacity', '0.75');
        svgOverlay.appendChild(line);

        const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        dot.setAttribute('id', c.dotId);
        dot.setAttribute('r', '2.5');
        dot.setAttribute('fill', c.color);
        svgOverlay.appendChild(dot);
      });
    }

    // Attach 3D reference dummy nodes to batteryGroup so they move with any rotation/panning
    spatialAnchors = configs.map(c => {
      const badgeEl = document.getElementById(c.id);
      const lineEl = document.getElementById(c.lineId);
      const dotEl = document.getElementById(c.dotId);

      const targetNode = new THREE.Object3D();
      targetNode.position.copy(c.targetPos);
      batteryGroup.add(targetNode);

      const anchorNode = new THREE.Object3D();
      anchorNode.position.copy(c.anchorPos);
      batteryGroup.add(anchorNode);

      return { badgeEl, lineEl, dotEl, targetNode, anchorNode };
    });
  }

  function updateSpatialLabels() {
    if (!camera || !renderer || spatialAnchors.length === 0) return;

    const width = container.clientWidth;
    const height = container.clientHeight;

    const tempTarget = new THREE.Vector3();
    const tempAnchor = new THREE.Vector3();

    spatialAnchors.forEach(({ badgeEl, lineEl, dotEl, targetNode, anchorNode }) => {
      if (!badgeEl) return;

      targetNode.getWorldPosition(tempTarget);
      anchorNode.getWorldPosition(tempAnchor);

      // Project 3D coordinates to Normalized Device Coordinates (-1 to +1)
      tempTarget.project(camera);
      tempAnchor.project(camera);

      // Behind camera check: hide if behind camera near plane
      if (tempAnchor.z > 1.0 || tempTarget.z > 1.0) {
        badgeEl.style.display = 'none';
        if (lineEl) lineEl.style.display = 'none';
        if (dotEl) dotEl.style.display = 'none';
        return;
      }

      // Convert NDC to canvas CSS pixel coordinates
      const targetX = (tempTarget.x * 0.5 + 0.5) * width;
      const targetY = (-tempTarget.y * 0.5 + 0.5) * height;

      const anchorX = (tempAnchor.x * 0.5 + 0.5) * width;
      const anchorY = (-tempAnchor.y * 0.5 + 0.5) * height;

      const badgeW = badgeEl.offsetWidth || 135;
      const badgeH = badgeEl.offsetHeight || 24;

      // Clamp label position within viewport boundaries with safety margins
      const badgeLeft = Math.max(10, Math.min(width - badgeW - 10, anchorX - badgeW / 2));
      const badgeTop = Math.max(10, Math.min(height - badgeH - 45, anchorY - badgeH));

      badgeEl.style.display = 'flex';
      badgeEl.style.transform = `translate(${badgeLeft}px, ${badgeTop}px)`;

      // Draw 1px leader line connecting badge bottom-center to exact 3D surface point
      const badgeBottomCenterX = badgeLeft + badgeW / 2;
      const badgeBottomCenterY = badgeTop + badgeH;

      if (lineEl) {
        lineEl.setAttribute('x1', badgeBottomCenterX);
        lineEl.setAttribute('y1', badgeBottomCenterY);
        lineEl.setAttribute('x2', targetX);
        lineEl.setAttribute('y2', targetY);
        lineEl.style.display = 'block';
      }

      if (dotEl) {
        dotEl.setAttribute('cx', targetX);
        dotEl.setAttribute('cy', targetY);
        dotEl.style.display = 'block';
      }
    });
  }

  // =========================================================================
  // 7. SIMULATION INTEGRATION LOOP
  // =========================================================================

  async function stepSimulation(delta) {
    if (!state.isRunning) return;

    state.timeElapsed += delta;

    const result = await dataProvider.fetchCycleStep({
      mode: state.mode,
      current: state.current,
      soc: state.soc,
      timeElapsed: state.timeElapsed,
      delta: delta
    });

    state.soc = result.soc_percent;
    state.voltage = result.voltage_V;
    state.seiThicknessNm = result.sei_thickness_nm;
    state.capacityLossAh = result.capacity_loss_ah;

    if (state.mode === 'charge' && state.voltage >= dataProvider.upperCutoffV) {
      state.isRunning = false;
      setEngineStatus('CUT-OFF // UPPER 4.2V REACHED', 'paused');
    } else if (state.mode === 'discharge' && state.voltage <= dataProvider.lowerCutoffV) {
      state.isRunning = false;
      setEngineStatus('CUT-OFF // LOWER 2.5V REACHED', 'paused');
    }

    updateTelemetryHUD();
  }

  function updateTelemetryHUD() {
    hudVoltage.innerHTML = `${state.voltage.toFixed(3)} <span class="dp-unit">V</span>`;
    hudSoc.innerHTML = `${state.soc.toFixed(1)} <span class="dp-unit">%</span>`;

    const signedCurrent = state.mode === 'charge' ? `+${state.current.toFixed(2)}` : (state.mode === 'discharge' ? `-${state.current.toFixed(2)}` : '0.00');
    hudCurrent.innerHTML = `${signedCurrent} <span class="dp-unit">A</span>`;

    hudSei.innerHTML = `${state.seiThicknessNm.toFixed(2)} <span class="dp-unit">nm</span>`;
    const qLossMah = (state.capacityLossAh * 1000.0).toFixed(1);
    hudQlli.innerHTML = `${qLossMah} <span class="dp-unit">mAh</span>`;

    const liveSoh = (100.0 - (state.capacityLossAh / dataProvider.nominalCapacityAh) * 100.0).toFixed(2);
    if (metricSohVal) {
      metricSohVal.textContent = `SOH: ${liveSoh}%`;
    }
  }

  function setEngineStatus(text, statusClass) {
    engineStatusLabel.textContent = text;
    engineStatusTag.className = `engine-status-tag ${statusClass || ''}`;
  }

  // =========================================================================
  // 8. EVENT HANDLERS & USER CONTROLS
  // =========================================================================

  function setupEventListeners() {
    btnModeIdle.addEventListener('click', () => setMode('idle'));
    btnModeCharge.addEventListener('click', () => setMode('charge'));
    btnModeDischarge.addEventListener('click', () => setMode('discharge'));

    sliderCurrent.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      inputCurrent.value = val.toFixed(1);
      setCurrent(val);
    });

    inputCurrent.addEventListener('change', (e) => {
      let val = parseFloat(e.target.value);
      val = Math.max(0.5, Math.min(10.0, isNaN(val) ? 5.0 : val));
      inputCurrent.value = val.toFixed(1);
      sliderCurrent.value = val;
      setCurrent(val);
    });

    function setCurrent(val) {
      state.current = val;
      const cRate = (val / dataProvider.nominalCapacityAh).toFixed(1);
      readoutCurrent.textContent = `${val.toFixed(2)} A (${cRate}C)`;
      updateTelemetryHUD();
    }

    btnRun.addEventListener('click', () => {
      state.isRunning = true;
      setEngineStatus('SIMULATION RUNNING', 'running');
    });

    btnPause.addEventListener('click', () => {
      state.isRunning = false;
      setEngineStatus('STANDBY // PAUSED', 'paused');
    });

    btnReset.addEventListener('click', () => {
      state.isRunning = false;
      state.soc = 65.0;
      state.timeElapsed = 0;
      state.seiThicknessNm = 5.0;
      state.capacityLossAh = 0.0;
      setMode('idle');
      setEngineStatus('STANDBY // RESET', '');
      updateTelemetryHUD();
    });

    btnRunLifecycle.addEventListener('click', async () => {
      setEngineStatus('SOLVING 100 CYCLES...', 'running');
      btnRunLifecycle.disabled = true;

      const results = await dataProvider.fetchLifecycleSimulation({
        n_cycles: 100,
        current: state.current
      });

      const finalSoh = results.SOH_percent[results.SOH_percent.length - 1];
      if (metricSohVal) {
        metricSohVal.textContent = `SOH: ${finalSoh.toFixed(2)}% (100 CYCLES)`;
      }
      setEngineStatus('100 CYCLES SOLVED', '');
      btnRunLifecycle.disabled = false;
    });

    btnCameraReset.addEventListener('click', () => {
      camera.position.set(0.0, 1.8, 5.0);
      controls.target.set(0, 0, 0);
      controls.update();
    });

    btnAutoRotate.addEventListener('click', () => {
      state.autoRotate = !state.autoRotate;
      btnAutoRotate.style.borderColor = state.autoRotate ? 'var(--accent-cyan)' : 'var(--border-strong)';
      btnAutoRotate.style.color = state.autoRotate ? 'var(--accent-cyan)' : 'var(--text-muted)';
    });
  }

  function setMode(newMode) {
    state.mode = newMode;

    modeButtons.forEach((btn) => {
      if (btn.getAttribute('data-mode') === newMode) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    });

    hudModePill.className = `hud-mode-pill ${newMode}`;
    if (newMode === 'idle') {
      hudModeText.textContent = 'MODE: IDLE (EQUILIBRIUM)';
    } else if (newMode === 'charge') {
      hudModeText.textContent = 'MODE: CHARGING CC (CATHODE -> ANODE)';
    } else if (newMode === 'discharge') {
      hudModeText.textContent = 'MODE: DISCHARGING CC (ANODE -> CATHODE)';
    }

    // Update dynamic transparency: light translucency (0.88) during charge/discharge, 100% solid in idle
    updateHousingTransparency(newMode);

    updateTelemetryHUD();
  }

  function onWindowResize() {
    if (!container || !renderer || !camera) return;
    const width = container.clientWidth;
    const height = container.clientHeight;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
  }

  // =========================================================================
  // 9. RENDER LOOP
  // =========================================================================

  function animate() {
    requestAnimationFrame(animate);

    const delta = clock.getDelta();

    // 1. Controls & optional auto-rotation
    if (controls) {
      controls.autoRotate = state.autoRotate;
      controls.autoRotateSpeed = 1.0;
      controls.update();
    }

    // 2. Update Li+ Particle Flow strictly inside calibrated bounds
    updateLithiumIons(delta);

    // 3. Project 3D Spatial Anchors to 2D Screen Badges & SVG Leader Lines
    updateSpatialLabels();

    // 4. Physical Step Integration
    stepSimulation(delta);

    // 5. Render WebGL Frame
    renderer.render(scene, camera);
  }

  // =========================================================================
  // 10. INITIALIZATION
  // =========================================================================

  function initApp() {
    initThreeScene();
    setupEventListeners();
    updateTelemetryHUD();
    animate();
    console.log('[BATT-SIM] LG M50T SPMe Digital Twin Workstation successfully initialized.');
  }

  if (document.readyState === 'loading') {
    window.addEventListener('DOMContentLoaded', initApp);
  } else {
    initApp();
  }

  window.batteryDigitalTwin = {
    state,
    dataProvider,
    calibration: VISUAL_CALIBRATION,
    setMode,
    start: () => btnRun.click(),
    pause: () => btnPause.click(),
    reset: () => btnReset.click(),
    runLifecycle: (cycles) => dataProvider.fetchLifecycleSimulation({ n_cycles: cycles || 100 })
  };

})();
