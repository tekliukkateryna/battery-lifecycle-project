/**
 * BATT-SIM 3D // Interactive Battery Simulation Engine
 * Three.js (r128), OrbitControls, GLTFLoader, DRACOLoader, Chart.js
 *
 * Updates:
 * - Direct loading from assets/battery.glb (with assets/45.glb & procedural fallbacks)
 * - Lithium-ion particle system (THREE.Points) flowing inside battery core (Cathode <-> Anode)
 * - Sleek traveling neon pulse / electric light along the model wire (no bulky spheres)
 * - Color & intensity linked to charge/discharge state & temperature overheat (>45°C)
 * - Full telemetry, Chart.js live updates, dark/light theme, and window resize support
 */

(function () {
  'use strict';

  // --- Simulation State ---
  const state = {
    mode: 'idle', // 'idle' | 'charge' | 'discharge'
    isRunning: false,
    soc: 65.0, // State of Charge (0 - 100 %)
    current: 2.5, // Amperes (0 - 10 A)
    temperature: 25.0, // Celsius (0 - 60 °C)
    voltage: 3.78, // Volts
    power: 9.45, // Watts
    internalResistance: 0.025, // Ohms
    capacityAh: 5.0, // 5000 mAh
    timeElapsed: 0, // seconds
    modelLoaded: false,
    modelSource: 'Ініціалізація...'
  };

  // --- DOM Elements ---
  const container = document.getElementById('canvas-container');
  const themeToggleBtn = document.getElementById('theme-toggle-btn');
  const simMainStatus = document.getElementById('sim-main-status');

  // Floating Status Elements
  const badgeMode = document.getElementById('badge-mode');
  const badgeModeText = document.getElementById('badge-mode-text');
  const badgeSource = document.getElementById('badge-source');
  const badgeSourceText = document.getElementById('badge-source-text');
  const floatingSocVal = document.getElementById('floating-soc-val');
  const chipVoltage = document.getElementById('chip-voltage');
  const chipCurrent = document.getElementById('chip-current');
  const chipTemp = document.getElementById('chip-temp');
  const chipPower = document.getElementById('chip-power');

  // Controls Elements
  const btnModeIdle = document.getElementById('btn-mode-idle');
  const btnModeCharge = document.getElementById('btn-mode-charge');
  const btnModeDischarge = document.getElementById('btn-mode-discharge');
  const modeButtons = [btnModeIdle, btnModeCharge, btnModeDischarge];

  const sliderCurrent = document.getElementById('slider-current');
  const valCurrent = document.getElementById('val-current');
  const sliderTemp = document.getElementById('slider-temp');
  const valTemp = document.getElementById('val-temp');

  const btnStart = document.getElementById('btn-start');
  const btnPause = document.getElementById('btn-pause');
  const btnReset = document.getElementById('btn-reset');
  const socBarText = document.getElementById('soc-bar-text');
  const socBarFill = document.getElementById('soc-bar-fill');
  const chartRateBadge = document.getElementById('chart-rate-badge');

  const btnCameraReset = document.getElementById('btn-camera-reset');
  const btnToggleWire = document.getElementById('btn-toggle-wire');

  // --- Three.js Variables ---
  let scene, camera, renderer, controls;
  let batteryGroup, batteryCoreMesh, batteryGlowMaterials = [];
  let ionParticleSystem, ionPositions, ionVelocities, ionInitialPositions;
  const ION_COUNT = 450;
  let wireGroup, wireCurve, wireBaseMesh, neonPulseBeams = [];
  const PULSE_COUNT = 5;
  let ambientLight, directionalLight, fillLight, corePointLight, wirePointLight;
  let clock = new THREE.Clock();

  // --- Chart.js Variables ---
  let telemetryChart;
  const chartMaxPoints = 35;
  let chartUpdateTimer = 0;

  // =========================================================================
  // 1. THREE.JS INITIALIZATION
  // =========================================================================
  function initThreeScene() {
    // 1. Scene
    scene = new THREE.Scene();
    updateSceneBackground();

    // 2. Camera
    const aspect = container.clientWidth / container.clientHeight;
    camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 100);
    camera.position.set(4.2, 3.0, 5.5);

    // 3. Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.25;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);

    // 4. OrbitControls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 2.2;
    controls.maxDistance = 14;
    controls.maxPolarAngle = Math.PI / 2 + 0.1;
    controls.target.set(0, 0, 0);

    // 5. Lighting
    setupLighting();

    // 6. Grid / Platform
    setupEnvironmentFloor();

    // 7. Battery Group
    batteryGroup = new THREE.Group();
    scene.add(batteryGroup);

    // 8. Lithium-Ion Particle System (Internal cell flow)
    setupLithiumIonSystem();

    // 9. Wire & Neon Pulse Effect (Along model connectors)
    setupModelWireCircuit();

    // 10. Load 3D Model from assets/battery.glb or Fallback
    loadBatteryModel();

    // 11. Window Resize Listener
    window.addEventListener('resize', onWindowResize);
  }

  function setupLighting() {
    ambientLight = new THREE.AmbientLight(0xffffff, 0.75);
    scene.add(ambientLight);

    directionalLight = new THREE.DirectionalLight(0xffffff, 1.35);
    directionalLight.position.set(6, 8, 6);
    directionalLight.castShadow = true;
    directionalLight.shadow.mapSize.width = 1024;
    directionalLight.shadow.mapSize.height = 1024;
    directionalLight.shadow.camera.near = 0.5;
    directionalLight.shadow.camera.far = 25;
    directionalLight.shadow.bias = -0.001;
    scene.add(directionalLight);

    fillLight = new THREE.DirectionalLight(0x00d9f5, 0.5);
    fillLight.position.set(-5, 3, -4);
    scene.add(fillLight);

    // Internal core glow light
    corePointLight = new THREE.PointLight(0x00f5a0, 1.4, 6.5);
    corePointLight.position.set(0, 0, 0);
    scene.add(corePointLight);

    // Wire pulse traveling highlight
    wirePointLight = new THREE.PointLight(0x00f5a0, 0.9, 3.5);
    wirePointLight.position.set(0, 1.25, 0);
    scene.add(wirePointLight);
  }

  function setupEnvironmentFloor() {
    // Tech base rings
    const ringGeo = new THREE.RingGeometry(1.6, 3.6, 64);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x00f5a0,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.12,
      wireframe: true
    });
    const platform = new THREE.Mesh(ringGeo, ringMat);
    platform.rotation.x = -Math.PI / 2;
    platform.position.y = -2.25;
    scene.add(platform);

    const baseDiskGeo = new THREE.CircleGeometry(4.2, 64);
    const baseDiskMat = new THREE.MeshStandardMaterial({
      color: 0x0a101d,
      roughness: 0.8,
      metalness: 0.3,
      transparent: true,
      opacity: 0.6
    });
    const baseDisk = new THREE.Mesh(baseDiskGeo, baseDiskMat);
    baseDisk.rotation.x = -Math.PI / 2;
    baseDisk.position.y = -2.26;
    baseDisk.receiveShadow = true;
    scene.add(baseDisk);
  }

  function updateSceneBackground() {
    const isDark = document.documentElement.getAttribute('data-theme') !== 'light';
    if (scene) {
      scene.background = null;
      scene.fog = new THREE.FogExp2(isDark ? 0x090d16 : 0xeef2f7, 0.05);
    }
  }

  // =========================================================================
  // 2. 3D MODEL LOADING (assets/battery.glb with DRACOLoader & Fallback)
  // =========================================================================
  function loadBatteryModel() {
    updateSourceBadge('Завантаження 3D...');

    const dracoLoader = new THREE.DRACOLoader();
    dracoLoader.setDecoderPath('https://www.gstatic.com/draco/versioned/decoders/1.4.1/');
    dracoLoader.setDecoderConfig({ type: 'js' });

    const gltfLoader = new THREE.GLTFLoader();
    gltfLoader.setDRACOLoader(dracoLoader);

    // Sequence of candidate files to try:
    // 1. assets/battery.glb (user requirement)
    // 2. assets/45.glb (original workspace asset)
    // 3. Fallback: Procedural Battery
    const candidates = ['assets/battery.glb', 'assets/45.glb'];

    function tryLoad(index) {
      if (index >= candidates.length) {
        console.warn('[Three.js] All GLB candidates failed. Creating Procedural Battery fallback.');
        createProceduralBattery();
        return;
      }

      const path = candidates[index];
      gltfLoader.load(
        path,
        function (gltf) {
          console.log('[Three.js] Model loaded successfully from:', path);
          processLoadedModel(gltf.scene, 'battery.glb (GLTF)');
        },
        function (xhr) {
          if (xhr.total > 0) {
            const percent = Math.round((xhr.loaded / xhr.total) * 100);
            updateSourceBadge(`Завантаження: ${percent}%`);
          }
        },
        function (err) {
          console.warn(`[Three.js] Failed to load ${path}:`, err);
          tryLoad(index + 1);
        }
      );
    }

    tryLoad(0);
  }

  function processLoadedModel(modelScene, sourceTitle) {
    batteryGroup.clear();
    batteryGlowMaterials = [];

    // Compute bounding box to normalize scale and center model
    const bbox = new THREE.Box3().setFromObject(modelScene);
    const center = new THREE.Vector3();
    const size = new THREE.Vector3();
    bbox.getCenter(center);
    bbox.getSize(size);

    modelScene.position.x = -center.x;
    modelScene.position.y = -center.y;
    modelScene.position.z = -center.z;

    const maxDim = Math.max(size.x, size.y, size.z);
    const targetScale = 3.8 / (maxDim || 1);
    modelScene.scale.set(targetScale, targetScale, targetScale);

    // Style model mesh: semi-translucent shell so lithium-ion flow inside is visible
    modelScene.traverse((node) => {
      if (node.isMesh) {
        node.castShadow = true;
        node.receiveShadow = true;

        if (node.material) {
          node.material = node.material.clone();
          if (node.material.isMeshStandardMaterial || node.material.isMeshPhysicalMaterial) {
            node.material.roughness = 0.28;
            node.material.metalness = 0.55;
            // High-tech translucent casing: allows viewing internal ion flow
            node.material.transparent = true;
            node.material.opacity = 0.82;
            node.material.depthWrite = true;
            node.material.emissive = new THREE.Color(0x00f5a0);
            node.material.emissiveIntensity = 0.15;
            batteryGlowMaterials.push(node.material);
          }
        }
      }
    });

    batteryGroup.add(modelScene);
    state.modelLoaded = true;
    updateSourceBadge(sourceTitle);
  }

  /**
   * Procedural Battery Fallback (CylinderGeometry)
   */
  function createProceduralBattery() {
    batteryGroup.clear();
    batteryGlowMaterials = [];

    const cellGroup = new THREE.Group();
    const labelTexture = createBatteryLabelCanvasTexture();

    const bodyRadius = 1.15;
    const bodyHeight = 3.6;
    const bodyGeo = new THREE.CylinderGeometry(bodyRadius, bodyRadius, bodyHeight, 64);
    const bodyMat = new THREE.MeshStandardMaterial({
      map: labelTexture,
      roughness: 0.32,
      metalness: 0.65,
      transparent: true,
      opacity: 0.85,
      emissive: new THREE.Color(0x00f5a0),
      emissiveIntensity: 0.15
    });
    const bodyMesh = new THREE.Mesh(bodyGeo, bodyMat);
    bodyMesh.castShadow = true;
    bodyMesh.receiveShadow = true;
    cellGroup.add(bodyMesh);
    batteryCoreMesh = bodyMesh;
    batteryGlowMaterials.push(bodyMat);

    const chromeMat = new THREE.MeshStandardMaterial({
      color: 0xd8e2ec,
      metalness: 0.92,
      roughness: 0.18
    });

    // Top cap & positive nub
    const topCapGeo = new THREE.CylinderGeometry(bodyRadius * 0.98, bodyRadius, 0.18, 64);
    const topCapMesh = new THREE.Mesh(topCapGeo, chromeMat);
    topCapMesh.position.y = bodyHeight / 2 + 0.09;
    cellGroup.add(topCapMesh);

    const nubGeo = new THREE.CylinderGeometry(0.38, 0.4, 0.28, 32);
    const positiveNub = new THREE.Mesh(nubGeo, chromeMat);
    positiveNub.position.y = bodyHeight / 2 + 0.28;
    cellGroup.add(positiveNub);

    // Bottom cap
    const bottomCapGeo = new THREE.CylinderGeometry(bodyRadius, bodyRadius * 0.96, 0.15, 64);
    const bottomCapMesh = new THREE.Mesh(bottomCapGeo, chromeMat);
    bottomCapMesh.position.y = -bodyHeight / 2 - 0.075;
    cellGroup.add(bottomCapMesh);

    batteryGroup.add(cellGroup);
    state.modelLoaded = true;
    updateSourceBadge('Процедурний циліндр (21700)');
  }

  function createBatteryLabelCanvasTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = 1024;
    canvas.height = 1024;
    const ctx = canvas.getContext('2d');

    const gradient = ctx.createLinearGradient(0, 0, 1024, 0);
    gradient.addColorStop(0, '#0c1524');
    gradient.addColorStop(0.3, '#192b45');
    gradient.addColorStop(0.7, '#192b45');
    gradient.addColorStop(1, '#0c1524');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 1024, 1024);

    ctx.fillStyle = '#00f5a0';
    ctx.fillRect(0, 150, 1024, 12);
    ctx.fillRect(0, 860, 1024, 12);

    ctx.fillStyle = 'rgba(0, 217, 245, 0.4)';
    ctx.fillRect(0, 175, 1024, 4);

    ctx.save();
    ctx.translate(512, 512);
    ctx.rotate(-Math.PI / 2);

    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 54px Outfit, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('LI-ION RECHARGEABLE CELL', 0, -40);

    ctx.fillStyle = '#00f5a0';
    ctx.font = 'bold 44px "JetBrains Mono", monospace';
    ctx.fillText('INR 21700-50E • 3.7V • 5000mAh', 0, 30);

    ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
    ctx.font = '28px "JetBrains Mono", monospace';
    ctx.fillText('+ POSITIVE TERMINAL  ▲  LITHIUM-ION CORE', 0, 95);
    ctx.restore();

    const texture = new THREE.CanvasTexture(canvas);
    return texture;
  }

  // =========================================================================
  // 3. LITHIUM-ION FLOW SYSTEM (Internal particle system: Cathode <-> Anode)
  // =========================================================================
  function setupLithiumIonSystem() {
    const geo = new THREE.BufferGeometry();
    ionPositions = new Float32Array(ION_COUNT * 3);
    ionVelocities = new Float32Array(ION_COUNT * 3);
    ionInitialPositions = new Float32Array(ION_COUNT * 3);
    const colors = new Float32Array(ION_COUNT * 3);

    // Boundary of internal cell active core
    // X axis: -1.6 (Cathode +) to +1.6 (Anode -)
    // Y axis: -1.1 to 0.45
    // Z axis: -0.65 to 0.65
    for (let i = 0; i < ION_COUNT; i++) {
      const idx = i * 3;
      const x = (Math.random() - 0.5) * 3.1;
      const y = (Math.random() - 0.5) * 1.5 - 0.3;
      const z = (Math.random() - 0.5) * 1.25;

      ionPositions[idx] = x;
      ionPositions[idx + 1] = y;
      ionPositions[idx + 2] = z;

      ionInitialPositions[idx] = x;
      ionInitialPositions[idx + 1] = y;
      ionInitialPositions[idx + 2] = z;

      // Slight random drift velocity
      ionVelocities[idx] = (Math.random() - 0.5) * 0.1;
      ionVelocities[idx + 1] = (Math.random() - 0.5) * 0.1;
      ionVelocities[idx + 2] = (Math.random() - 0.5) * 0.1;

      // Initial color (cyan neon)
      colors[idx] = 0.0;
      colors[idx + 1] = 0.95;
      colors[idx + 2] = 0.75;
    }

    geo.setAttribute('position', new THREE.BufferAttribute(ionPositions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const ionTexture = createIonPointTexture();
    const mat = new THREE.PointsMaterial({
      size: 0.16,
      map: ionTexture,
      vertexColors: true,
      transparent: true,
      opacity: 0.88,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });

    ionParticleSystem = new THREE.Points(geo, mat);
    scene.add(ionParticleSystem);
  }

  function createIonPointTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = 64;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');

    const grad = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
    grad.addColorStop(0, 'rgba(255, 255, 255, 1)');
    grad.addColorStop(0.25, 'rgba(0, 245, 160, 0.95)');
    grad.addColorStop(0.6, 'rgba(0, 217, 245, 0.45)');
    grad.addColorStop(1, 'rgba(0, 0, 0, 0)');

    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 64, 64);

    const texture = new THREE.CanvasTexture(canvas);
    return texture;
  }

  function updateLithiumIons(delta) {
    if (!ionParticleSystem) return;

    const positions = ionParticleSystem.geometry.attributes.position.array;
    const colors = ionParticleSystem.geometry.attributes.color.array;
    const time = clock.getElapsedTime();

    // Determine direction and drift speed:
    // Charge: Cathode (-1.55) -> Anode (+1.55) => direction = +1
    // Discharge: Anode (+1.55) -> Cathode (-1.55) => direction = -1
    // Idle: direction = 0 (only micro Brownian oscillation)
    let direction = 0;
    let baseSpeed = 0;

    if (state.isRunning) {
      if (state.mode === 'charge') {
        direction = 1;
        baseSpeed = (0.2 + (state.current / 10) * 1.4);
      } else if (state.mode === 'discharge') {
        direction = -1;
        baseSpeed = (0.2 + (state.current / 10) * 1.4);
      }
    }

    // Color interpolation based on state & temperature
    const targetColor = new THREE.Color();
    if (state.mode === 'charge') {
      targetColor.set(0x00d9f5); // Bright electric cyan
    } else if (state.mode === 'discharge') {
      targetColor.set(0xffb703); // Warm amber
    } else {
      targetColor.set(0x00f5a0); // Calm mint green
    }

    // Temperature overheat warning (>45°C) -> red
    if (state.temperature > 45) {
      targetColor.lerp(new THREE.Color(0xff2244), (state.temperature - 45) / 15);
    }

    const minX = -1.55;
    const maxX = 1.55;
    const rangeX = maxX - minX;

    for (let i = 0; i < ION_COUNT; i++) {
      const idx = i * 3;

      if (direction !== 0) {
        // Directed Li+ migration
        positions[idx] += direction * baseSpeed * delta;

        // Wrap around at cell boundaries
        if (positions[idx] > maxX) positions[idx] -= rangeX;
        if (positions[idx] < minX) positions[idx] += rangeX;
      } else {
        // Idle gentle thermal vibration
        positions[idx] = ionInitialPositions[idx] + Math.sin(time * 3 + i) * 0.04;
      }

      // Vertical & depth micro-fluctuations (electrolyte turbulence)
      positions[idx + 1] = ionInitialPositions[idx + 1] + Math.sin(time * 4 + i * 0.5) * 0.05;
      positions[idx + 2] = ionInitialPositions[idx + 2] + Math.cos(time * 3.5 + i * 0.7) * 0.04;

      // Particle color
      colors[idx] = targetColor.r;
      colors[idx + 1] = targetColor.g;
      colors[idx + 2] = targetColor.b;
    }

    ionParticleSystem.geometry.attributes.position.needsUpdate = true;
    ionParticleSystem.geometry.attributes.color.needsUpdate = true;
  }

  // =========================================================================
  // 4. MODEL WIRE CIRCUIT & TRAVELING NEON PULSE EFFECT
  // =========================================================================
  function setupModelWireCircuit() {
    wireGroup = new THREE.Group();

    // Precise 3D trajectory matching the top terminal arch connector on the battery model
    wireCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(-1.35, 0.52, 0.0),
      new THREE.Vector3(-1.15, 1.12, 0.0),
      new THREE.Vector3(-0.6, 1.28, 0.0),
      new THREE.Vector3(0.0, 1.32, 0.0),
      new THREE.Vector3(0.6, 1.28, 0.0),
      new THREE.Vector3(1.15, 1.12, 0.0),
      new THREE.Vector3(1.35, 0.52, 0.0)
    ]);

    // Sleek thin base wire core
    const tubeGeo = new THREE.TubeGeometry(wireCurve, 64, 0.024, 8, false);
    const tubeMat = new THREE.MeshStandardMaterial({
      color: 0x1a2638,
      metalness: 0.9,
      roughness: 0.25,
      transparent: true,
      opacity: 0.75
    });
    wireBaseMesh = new THREE.Mesh(tubeGeo, tubeMat);
    wireGroup.add(wireBaseMesh);

    // Traveling Neon Pulses (Electric impulses running along the wire)
    // Replaces bulky floating spheres with sleek high-tech traveling light segments
    const pulseSegmentGeo = new THREE.SphereGeometry(0.048, 12, 12);

    for (let i = 0; i < PULSE_COUNT; i++) {
      const pulseMat = new THREE.MeshBasicMaterial({
        color: 0x00f5a0,
        transparent: true,
        opacity: 0.95
      });
      const pulseMesh = new THREE.Mesh(pulseSegmentGeo, pulseMat);

      // Trailing glow bead
      const trailMat = new THREE.MeshBasicMaterial({
        color: 0x00d9f5,
        transparent: true,
        opacity: 0.4
      });
      const trailMesh = new THREE.Mesh(new THREE.SphereGeometry(0.032, 8, 8), trailMat);
      pulseMesh.add(trailMesh);
      trailMesh.position.set(0, 0, 0);

      neonPulseBeams.push({
        mesh: pulseMesh,
        trail: trailMesh,
        material: pulseMat,
        trailMaterial: trailMat,
        t: i / PULSE_COUNT
      });

      wireGroup.add(pulseMesh);
    }

    scene.add(wireGroup);
  }

  function updateNeonWirePulse(delta) {
    if (!wireCurve || neonPulseBeams.length === 0) return;

    let speed = 0;
    let flowDirection = 0;
    let pulseColor = new THREE.Color(0x00f5a0);

    if (state.isRunning) {
      if (state.mode === 'charge') {
        flowDirection = -1; // Current flows inward
        speed = 0.25 + (state.current / 10) * 0.9;
        pulseColor.set(0x00d9f5); // Neon cyan
      } else if (state.mode === 'discharge') {
        flowDirection = 1; // Current flows outward
        speed = 0.25 + (state.current / 10) * 0.9;
        pulseColor.set(0xffb703); // Energetic amber
      }
    }

    // Heat override (>45°C) -> danger hot red
    if (state.temperature > 45) {
      pulseColor.lerp(new THREE.Color(0xff2244), (state.temperature - 45) / 15);
    }

    neonPulseBeams.forEach((pulse, idx) => {
      if (speed > 0) {
        pulse.t += flowDirection * speed * delta;
        if (pulse.t > 1) pulse.t -= 1;
        if (pulse.t < 0) pulse.t += 1;
      }

      const pos = wireCurve.getPointAt(pulse.t);
      pulse.mesh.position.copy(pos);

      // Trailing position slightly offset behind direction of motion
      const trailT = Math.max(0, Math.min(1, pulse.t - flowDirection * 0.035));
      const trailPos = wireCurve.getPointAt(trailT);
      pulse.trail.position.copy(trailPos).sub(pos);

      // Color and pulsation
      pulse.material.color.copy(pulseColor);
      pulse.trailMaterial.color.copy(pulseColor);

      const pulseScale = state.isRunning ? 1.0 + 0.35 * Math.sin(clock.getElapsedTime() * 12 + idx) : 0.6;
      pulse.mesh.scale.set(pulseScale, pulseScale, pulseScale);
      pulse.mesh.visible = state.isRunning || state.mode !== 'idle';
    });

    if (wirePointLight) {
      wirePointLight.color.copy(pulseColor);
      wirePointLight.intensity = state.isRunning ? 0.8 + (state.current / 10) * 0.8 : 0.2;
    }
  }

  // =========================================================================
  // 5. BATTERY SIMULATION DYNAMICS & EMISSIVE GLOW
  // =========================================================================
  function updateSimulation(delta) {
    if (state.isRunning) {
      state.timeElapsed += delta;

      // Accelerated interactive factor
      const speedUp = 35.0;
      const socChangePerSec = (state.current / (state.capacityAh * 3600)) * 100 * speedUp;

      if (state.mode === 'charge') {
        state.soc = Math.min(100.0, state.soc + socChangePerSec * delta);
        if (state.soc >= 100.0) {
          state.isRunning = false;
          updateStatusPill('ЗАРЯД ЗАВЕРШЕНО (100%)', 'var(--glow-green)');
        }
      } else if (state.mode === 'discharge') {
        state.soc = Math.max(0.0, state.soc - socChangePerSec * delta);
        if (state.soc <= 0.0) {
          state.isRunning = false;
          updateStatusPill('РОЗРЯДЖЕНО (0%)', 'var(--glow-red)');
        }
      }

      // Self-heating effect if current is high
      if (state.current > 4.0) {
        const tempIncrease = ((state.current - 4.0) * 0.02) * delta;
        state.temperature = Math.min(60.0, state.temperature + tempIncrease);
        sliderTemp.value = state.temperature.toFixed(1);
        valTemp.textContent = `${state.temperature.toFixed(1)} °C`;
      }
    }

    // Dynamic OCV (Open Circuit Voltage) Curve + IR drop
    const ocv = 3.0 + 1.15 * (state.soc / 100) + 0.05 * Math.sqrt(state.soc / 100);
    const irDrop = state.current * state.internalResistance;

    if (state.mode === 'charge' && state.isRunning) {
      state.voltage = ocv + irDrop;
    } else if (state.mode === 'discharge' && state.isRunning) {
      state.voltage = Math.max(2.8, ocv - irDrop);
    } else {
      state.voltage = ocv;
    }

    state.power = state.voltage * (state.isRunning && state.mode !== 'idle' ? state.current : 0);

    // Update Materials Emissive Glow
    updateMaterialsGlow();

    // Update UI Badges & Numbers
    updateTelemetryUI();

    // Record Telemetry Data periodically
    chartUpdateTimer += delta;
    if (chartUpdateTimer >= 0.4) {
      chartUpdateTimer = 0;
      recordTelemetryData();
    }
  }

  function updateMaterialsGlow() {
    const time = clock.getElapsedTime();
    let glowColor = new THREE.Color(0x00f5a0);
    let intensity = 0.2;

    if (state.mode === 'charge') {
      glowColor.set(0x00d9f5);
      intensity = state.isRunning ? 0.75 + 0.4 * Math.sin(time * 4) : 0.35;
    } else if (state.mode === 'discharge') {
      glowColor.set(0xff5500);
      intensity = state.isRunning ? 0.85 + 0.45 * Math.sin(time * 5) : 0.35;
    } else {
      glowColor.set(0x00f5a0);
      intensity = 0.22 + 0.08 * Math.sin(time * 2);
    }

    // High temperature warning shift to red
    if (state.temperature > 45) {
      glowColor.lerp(new THREE.Color(0xff1144), (state.temperature - 45) / 15);
    }

    batteryGlowMaterials.forEach((mat) => {
      mat.emissive.copy(glowColor);
      mat.emissiveIntensity = intensity;
    });

    if (corePointLight) {
      corePointLight.color.copy(glowColor);
      corePointLight.intensity = intensity * 1.8;
    }
  }

  function updateTelemetryUI() {
    floatingSocVal.textContent = `${state.soc.toFixed(1)}%`;
    socBarText.textContent = `${state.soc.toFixed(1)}%`;
    socBarFill.style.width = `${Math.min(100, Math.max(0, state.soc))}%`;

    chipVoltage.textContent = `${state.voltage.toFixed(2)} V`;
    chipCurrent.textContent = `${state.current.toFixed(1)} A`;
    chipTemp.textContent = `${state.temperature.toFixed(1)} °C`;
    chipPower.textContent = `${state.power.toFixed(1)} W`;

    if (state.soc < 20) {
      socBarFill.style.background = 'linear-gradient(135deg, #ff3366, #ff6b8b)';
    } else if (state.soc < 50) {
      socBarFill.style.background = 'linear-gradient(135deg, #ffb703, #fb8500)';
    } else {
      socBarFill.style.background = 'var(--accent-gradient)';
    }
  }

  function updateStatusPill(text, color) {
    const pillText = simMainStatus.querySelector('.status-text');
    const dot = simMainStatus.querySelector('.status-dot');
    if (pillText) pillText.textContent = text;
    if (dot && color) {
      dot.style.backgroundColor = color;
      dot.style.boxShadow = `0 0 10px ${color}`;
    }
  }

  function updateSourceBadge(text) {
    if (badgeSourceText) {
      badgeSourceText.textContent = text;
    }
  }

  // =========================================================================
  // 6. CHART.JS TELEMETRY GRAPH
  // =========================================================================
  function initChart() {
    const ctx = document.getElementById('telemetryChart').getContext('2d');
    const isDark = document.documentElement.getAttribute('data-theme') !== 'light';

    const gridColor = isDark ? 'rgba(255, 255, 255, 0.06)' : 'rgba(0, 0, 0, 0.06)';
    const textColor = isDark ? '#94a3b8' : '#475569';

    const initialLabels = [];
    const initialData = [];
    for (let i = 15; i >= 0; i--) {
      initialLabels.push(`-${i}с`);
      initialData.push(state.soc);
    }

    telemetryChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: initialLabels,
        datasets: [
          {
            label: 'Рівень заряду SoC (%)',
            data: initialData,
            borderColor: '#00f5a0',
            backgroundColor: 'rgba(0, 245, 160, 0.08)',
            borderWidth: 2.5,
            pointRadius: 0,
            pointHoverRadius: 5,
            fill: true,
            tension: 0.35
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 300 },
        scales: {
          x: {
            grid: { color: gridColor },
            ticks: { color: textColor, font: { family: 'JetBrains Mono', size: 10 } }
          },
          y: {
            min: 0,
            max: 100,
            grid: { color: gridColor },
            ticks: {
              color: textColor,
              font: { family: 'JetBrains Mono', size: 10 },
              callback: function (val) {
                return val + '%';
              }
            }
          }
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: isDark ? 'rgba(15, 22, 38, 0.9)' : 'rgba(255, 255, 255, 0.95)',
            titleColor: isDark ? '#f8fafc' : '#0f172a',
            bodyColor: '#00f5a0',
            borderColor: 'rgba(0, 245, 160, 0.3)',
            borderWidth: 1,
            titleFont: { family: 'Outfit', size: 12 },
            bodyFont: { family: 'JetBrains Mono', size: 12, weight: 'bold' }
          }
        }
      }
    });
  }

  function recordTelemetryData() {
    if (!telemetryChart) return;

    const currentTimeLabel = `${Math.round(state.timeElapsed)}с`;
    telemetryChart.data.labels.push(currentTimeLabel);
    telemetryChart.data.datasets[0].data.push(Number(state.soc.toFixed(1)));

    if (telemetryChart.data.labels.length > chartMaxPoints) {
      telemetryChart.data.labels.shift();
      telemetryChart.data.datasets[0].data.shift();
    }

    if (state.mode === 'charge') {
      telemetryChart.data.datasets[0].borderColor = '#00d9f5';
      telemetryChart.data.datasets[0].backgroundColor = 'rgba(0, 217, 245, 0.1)';
      chartRateBadge.textContent = `+${((state.current * 0.2)).toFixed(2)} %/с`;
      chartRateBadge.style.color = 'var(--glow-blue)';
    } else if (state.mode === 'discharge') {
      telemetryChart.data.datasets[0].borderColor = '#ffb703';
      telemetryChart.data.datasets[0].backgroundColor = 'rgba(255, 183, 3, 0.1)';
      chartRateBadge.textContent = `-${((state.current * 0.2)).toFixed(2)} %/с`;
      chartRateBadge.style.color = 'var(--glow-yellow)';
    } else {
      telemetryChart.data.datasets[0].borderColor = '#00f5a0';
      telemetryChart.data.datasets[0].backgroundColor = 'rgba(0, 245, 160, 0.08)';
      chartRateBadge.textContent = '0.00 %/с';
      chartRateBadge.style.color = 'var(--accent-color)';
    }

    telemetryChart.update('none');
  }

  function updateChartTheme() {
    if (!telemetryChart) return;
    const isDark = document.documentElement.getAttribute('data-theme') !== 'light';
    const gridColor = isDark ? 'rgba(255, 255, 255, 0.06)' : 'rgba(0, 0, 0, 0.06)';
    const textColor = isDark ? '#94a3b8' : '#475569';

    telemetryChart.options.scales.x.grid.color = gridColor;
    telemetryChart.options.scales.x.ticks.color = textColor;
    telemetryChart.options.scales.y.grid.color = gridColor;
    telemetryChart.options.scales.y.ticks.color = textColor;
    telemetryChart.update();
  }

  // =========================================================================
  // 7. EVENT LISTENERS & USER CONTROLS
  // =========================================================================
  function setupEventListeners() {
    // Mode Buttons
    btnModeIdle.addEventListener('click', () => setMode('idle'));
    btnModeCharge.addEventListener('click', () => setMode('charge'));
    btnModeDischarge.addEventListener('click', () => setMode('discharge'));

    // Range Sliders
    sliderCurrent.addEventListener('input', (e) => {
      state.current = parseFloat(e.target.value);
      valCurrent.textContent = `${state.current.toFixed(1)} A`;
    });

    sliderTemp.addEventListener('input', (e) => {
      state.temperature = parseFloat(e.target.value);
      valTemp.textContent = `${state.temperature.toFixed(1)} °C`;
    });

    // Simulation Action Buttons
    btnStart.addEventListener('click', () => {
      state.isRunning = true;
      updateStatusPill('СИМУЛЯЦІЯ: АКТИВНА', 'var(--accent-color)');
      btnStart.classList.add('active');
      btnPause.classList.remove('active');
    });

    btnPause.addEventListener('click', () => {
      state.isRunning = false;
      updateStatusPill('СИМУЛЯЦІЯ: ПАУЗА', 'var(--glow-yellow)');
      btnPause.classList.add('active');
      btnStart.classList.remove('active');
    });

    btnReset.addEventListener('click', () => {
      state.isRunning = false;
      state.soc = 65.0;
      state.timeElapsed = 0;
      setMode('idle');
      updateStatusPill('СИМУЛЯЦІЯ: СКИНУТО', 'var(--accent-color)');

      if (telemetryChart) {
        telemetryChart.data.labels = [];
        telemetryChart.data.datasets[0].data = [];
        for (let i = 15; i >= 0; i--) {
          telemetryChart.data.labels.push(`-${i}с`);
          telemetryChart.data.datasets[0].data.push(65.0);
        }
        telemetryChart.update();
      }
      updateTelemetryUI();
    });

    // Theme Toggle
    themeToggleBtn.addEventListener('click', toggleTheme);

    // Camera Reset
    btnCameraReset.addEventListener('click', () => {
      camera.position.set(4.2, 3.0, 5.5);
      controls.target.set(0, 0, 0);
      controls.update();
    });

    // Toggle Wire Visibility
    let wireVisible = true;
    btnToggleWire.addEventListener('click', () => {
      wireVisible = !wireVisible;
      if (wireGroup) wireGroup.visible = wireVisible;
      btnToggleWire.style.opacity = wireVisible ? '1' : '0.5';
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

    badgeMode.className = `badge mode-badge mode-${newMode}`;
    if (newMode === 'idle') {
      badgeModeText.textContent = 'СПОКІЙ';
    } else if (newMode === 'charge') {
      badgeModeText.textContent = 'ЗАРЯДЖАННЯ';
    } else if (newMode === 'discharge') {
      badgeModeText.textContent = 'РОЗРЯДЖАННЯ';
    }
  }

  function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', newTheme);
    updateSceneBackground();
    updateChartTheme();
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
  // 8. ANIMATION RENDER LOOP
  // =========================================================================
  function animate() {
    requestAnimationFrame(animate);

    const delta = clock.getDelta();

    // Subtle gentle cell rotation
    if (batteryGroup) {
      batteryGroup.rotation.y += (state.isRunning ? 0.25 : 0.08) * delta;
    }

    controls.update();

    // 1. Update Lithium-Ion particle flow inside the cell
    updateLithiumIons(delta);

    // 2. Update Traveling Neon Pulse along model wire
    updateNeonWirePulse(delta);

    // 3. Physics & Simulation updates
    updateSimulation(delta);

    // Render frame
    renderer.render(scene, camera);
  }

  // =========================================================================
  // 9. INITIALIZE APPLICATION
  // =========================================================================
  function startApp() {
    initThreeScene();
    initChart();
    setupEventListeners();
    animate();
    console.log('[BATT-SIM 3D] Application initialized successfully with Lithium-ion particle system.');
  }

  if (document.readyState === 'loading') {
    window.addEventListener('DOMContentLoaded', startApp);
  } else {
    startApp();
  }

  // Expose API on window for testing & programmatic control
  window.battSim = {
    state,
    setMode,
    start: () => {
      state.isRunning = true;
      updateStatusPill('СИМУЛЯЦІЯ: АКТИВНА', 'var(--accent-color)');
      btnStart.classList.add('active');
      btnPause.classList.remove('active');
    },
    pause: () => {
      state.isRunning = false;
      updateStatusPill('СИМУЛЯЦІЯ: ПАУЗА', 'var(--glow-yellow)');
      btnPause.classList.add('active');
      btnStart.classList.remove('active');
    },
    reset: () => {
      btnReset.click();
    },
    toggleTheme
  };
})();
