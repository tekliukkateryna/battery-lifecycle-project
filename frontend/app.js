/**
 * BATT-SIM 3D // Interactive Battery Simulation Engine
 * Three.js (r128), OrbitControls, GLTFLoader, DRACOLoader, Chart.js
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
  let wireCurve, wireMesh, electronParticles = [];
  let wireGroup;
  let ambientLight, directionalLight, fillLight, corePointLight;
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
    camera.position.set(4.5, 3.2, 5.8);

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
    controls.minDistance = 2.5;
    controls.maxDistance = 14;
    controls.maxPolarAngle = Math.PI / 2 + 0.1; // Prevent going too far below floor
    controls.target.set(0, 0, 0);

    // 5. Lighting
    setupLighting();

    // 6. Grid / Floor reflection platform
    setupEnvironmentFloor();

    // 7. Battery Group
    batteryGroup = new THREE.Group();
    scene.add(batteryGroup);

    // 8. Wire & Electrons Circuit
    setupElectricalCircuit();

    // 9. Load Model with GLTFLoader/DRACOLoader or Fallback
    loadBatteryModel();

    // 10. Window Resize Listener
    window.addEventListener('resize', onWindowResize);
  }

  function setupLighting() {
    ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambientLight);

    directionalLight = new THREE.DirectionalLight(0xffffff, 1.4);
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

    // Dynamic glowing point light inside/around the battery
    corePointLight = new THREE.PointLight(0x00f5a0, 1.2, 7);
    corePointLight.position.set(0, 0, 0);
    scene.add(corePointLight);
  }

  function setupEnvironmentFloor() {
    // Subtle circular technological platform beneath the cell
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
      scene.background = null; // Canvas transparent, CSS controls outer look or fog
      scene.fog = new THREE.FogExp2(isDark ? 0x090d16 : 0xeef2f7, 0.05);
    }
  }

  // =========================================================================
  // 2. 3D MODEL LOADING & DRACO / FALLBACK ENGINE
  // =========================================================================
  function loadBatteryModel() {
    updateSourceBadge('Завантаження 3D...');

    // Setup DRACOLoader
    const dracoLoader = new THREE.DRACOLoader();
    dracoLoader.setDecoderPath('https://www.gstatic.com/draco/versioned/decoders/1.4.1/');
    dracoLoader.setDecoderConfig({ type: 'js' });

    // Setup GLTFLoader
    const gltfLoader = new THREE.GLTFLoader();
    gltfLoader.setDRACOLoader(dracoLoader);

    // List of candidate files to attempt loading:
    // Prompt asks to configure GLTF/DRACO and attempt loading model from assets/shaded.jpg (or .glb/.gltf)
    // We try 'assets/model (1).glb' first, then 'assets/shaded.jpg', and if all fail, use fallback procedural cylinder.
    const primaryCandidate = 'assets/model (1).glb';

    gltfLoader.load(
      primaryCandidate,
      function (gltf) {
        // Success callback
        console.log('[Three.js] GLTF model loaded successfully from:', primaryCandidate);
        processLoadedModel(gltf.scene, 'GLTF 3D Asset');
      },
      function (xhr) {
        // Progress callback
        if (xhr.total > 0) {
          const percent = Math.round((xhr.loaded / xhr.total) * 100);
          updateSourceBadge(`Завантаження: ${percent}%`);
        }
      },
      function (error) {
        // On error, test second candidate: assets/shaded.jpg
        console.warn('[Three.js] Primary GLTF failed or not found, attempting assets/shaded.jpg...', error);
        
        gltfLoader.load(
          'assets/shaded.jpg',
          function (gltf2) {
            console.log('[Three.js] Loaded model from shaded.jpg successfully');
            processLoadedModel(gltf2.scene, 'GLTF (shaded)');
          },
          null,
          function (err2) {
            console.warn('[Three.js] Fallback triggered! Creating Procedural Battery (CylinderGeometry).', err2);
            createProceduralBattery();
          }
        );
      }
    );
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

    // Center model at origin
    modelScene.position.x = -center.x;
    modelScene.position.y = -center.y;
    modelScene.position.z = -center.z;

    const maxDim = Math.max(size.x, size.y, size.z);
    const targetScale = 4.0 / (maxDim || 1);
    modelScene.scale.set(targetScale, targetScale, targetScale);

    // Traverse meshes to configure materials & glowing effects
    modelScene.traverse((node) => {
      if (node.isMesh) {
        node.castShadow = true;
        node.receiveShadow = true;

        if (node.material) {
          // Clone material to avoid modifying shared cache
          node.material = node.material.clone();
          if (node.material.isMeshStandardMaterial || node.material.isMeshPhysicalMaterial) {
            node.material.roughness = Math.min(node.material.roughness, 0.45);
            node.material.metalness = Math.max(node.material.metalness, 0.5);
            node.material.emissive = new THREE.Color(0x00f5a0);
            node.material.emissiveIntensity = 0.2;
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
   * High-fidelity Procedural Battery (CylinderGeometry Fallback)
   * Ensures the application functions reliably even if external files are missing.
   */
  function createProceduralBattery() {
    batteryGroup.clear();
    batteryGlowMaterials = [];

    const cellGroup = new THREE.Group();

    // 1. Procedural High-Tech Label Texture
    const labelTexture = createBatteryLabelCanvasTexture();

    // 2. Main Battery Cylinder Body
    const bodyRadius = 1.15;
    const bodyHeight = 3.8;
    const bodyGeo = new THREE.CylinderGeometry(bodyRadius, bodyRadius, bodyHeight, 64);
    
    const bodyMat = new THREE.MeshStandardMaterial({
      map: labelTexture,
      roughness: 0.32,
      metalness: 0.65,
      emissive: new THREE.Color(0x00f5a0),
      emissiveIntensity: 0.15
    });
    const bodyMesh = new THREE.Mesh(bodyGeo, bodyMat);
    bodyMesh.castShadow = true;
    bodyMesh.receiveShadow = true;
    cellGroup.add(bodyMesh);
    batteryCoreMesh = bodyMesh;
    batteryGlowMaterials.push(bodyMat);

    // 3. Metallic Top Crimp & Cap Ring
    const topCapGeo = new THREE.CylinderGeometry(bodyRadius * 0.98, bodyRadius, 0.18, 64);
    const chromeMat = new THREE.MeshStandardMaterial({
      color: 0xd8e2ec,
      metalness: 0.92,
      roughness: 0.18
    });
    const topCapMesh = new THREE.Mesh(topCapGeo, chromeMat);
    topCapMesh.position.y = bodyHeight / 2 + 0.09;
    topCapMesh.castShadow = true;
    cellGroup.add(topCapMesh);

    // 4. Positive Terminal Nub (+)
    const nubGeo = new THREE.CylinderGeometry(0.38, 0.4, 0.28, 32);
    const positiveNub = new THREE.Mesh(nubGeo, chromeMat);
    positiveNub.position.y = bodyHeight / 2 + 0.28;
    positiveNub.castShadow = true;
    cellGroup.add(positiveNub);

    // Insulator Ring around positive nub
    const insulatorGeo = new THREE.CylinderGeometry(0.65, 0.65, 0.06, 32);
    const insulatorMat = new THREE.MeshStandardMaterial({
      color: 0x1e293b,
      roughness: 0.8,
      metalness: 0.1
    });
    const insulatorMesh = new THREE.Mesh(insulatorGeo, insulatorMat);
    insulatorMesh.position.y = bodyHeight / 2 + 0.16;
    cellGroup.add(insulatorMesh);

    // 5. Negative Terminal Bottom Flat Plate (-)
    const bottomCapGeo = new THREE.CylinderGeometry(bodyRadius, bodyRadius * 0.96, 0.15, 64);
    const bottomCapMesh = new THREE.Mesh(bottomCapGeo, chromeMat);
    bottomCapMesh.position.y = -bodyHeight / 2 - 0.075;
    bottomCapMesh.receiveShadow = true;
    cellGroup.add(bottomCapMesh);

    // 6. Glowing Energy Level Ring (Indicator Band)
    const ringGeo = new THREE.TorusGeometry(bodyRadius + 0.015, 0.045, 16, 64);
    const ringMat = new THREE.MeshStandardMaterial({
      color: 0x00f5a0,
      emissive: new THREE.Color(0x00f5a0),
      emissiveIntensity: 1.5,
      roughness: 0.2,
      metalness: 0.1
    });
    const ringMesh = new THREE.Mesh(ringGeo, ringMat);
    ringMesh.rotation.x = Math.PI / 2;
    ringMesh.position.y = 1.0;
    cellGroup.add(ringMesh);
    batteryGlowMaterials.push(ringMat);

    // Second status ring
    const ring2 = ringMesh.clone();
    ring2.position.y = -1.0;
    cellGroup.add(ring2);
    batteryGlowMaterials.push(ring2.material);

    batteryGroup.add(cellGroup);
    state.modelLoaded = true;
    updateSourceBadge('Процедурний Циліндр (21700)');
  }

  function createBatteryLabelCanvasTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = 1024;
    canvas.height = 1024;
    const ctx = canvas.getContext('2d');

    // Background gradient for outer sleeve
    const gradient = ctx.createLinearGradient(0, 0, 1024, 0);
    gradient.addColorStop(0, '#0c1524');
    gradient.addColorStop(0.3, '#192b45');
    gradient.addColorStop(0.7, '#192b45');
    gradient.addColorStop(1, '#0c1524');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 1024, 1024);

    // Accent technical stripes
    ctx.fillStyle = '#00f5a0';
    ctx.fillRect(0, 150, 1024, 12);
    ctx.fillRect(0, 860, 1024, 12);

    ctx.fillStyle = 'rgba(0, 217, 245, 0.4)';
    ctx.fillRect(0, 175, 1024, 4);

    // Micro grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let x = 0; x < 1024; x += 32) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, 1024);
      ctx.stroke();
    }

    // Typography & Battery Specs
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
    ctx.fillText('+ POSITIVE TERMINAL  ▲  DO NOT CRUSH OR HEAT', 0, 95);

    ctx.restore();

    const texture = new THREE.CanvasTexture(canvas);
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.ClampToEdgeWrapping;
    return texture;
  }

  // =========================================================================
  // 3. WIRE & ELECTRON CIRCUIT SIMULATION
  // =========================================================================
  function setupElectricalCircuit() {
    wireGroup = new THREE.Group();

    // 1. CatmullRomCurve3: 3D path looping gracefully from positive terminal to negative
    wireCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0, 2.2, 0),        // Start at top terminal (+)
      new THREE.Vector3(0.5, 2.5, 0.3),
      new THREE.Vector3(2.2, 2.7, 0.8),    // Loop out into space
      new THREE.Vector3(3.4, 1.2, 0.4),    // Through circuit side
      new THREE.Vector3(3.2, -1.0, -0.4),  // Lower quadrant
      new THREE.Vector3(2.0, -2.4, -0.6),
      new THREE.Vector3(0.5, -2.4, -0.2),
      new THREE.Vector3(0, -2.1, 0)        // End at bottom terminal (-)
    ]);

    // 2. 3D Wire Tube Geometry
    const tubeGeo = new THREE.TubeGeometry(wireCurve, 80, 0.042, 12, false);
    const tubeMat = new THREE.MeshStandardMaterial({
      color: 0x223344,
      metalness: 0.85,
      roughness: 0.25,
      transparent: true,
      opacity: 0.85
    });
    wireMesh = new THREE.Mesh(tubeGeo, tubeMat);
    wireGroup.add(wireMesh);

    // 3. Electron Swarm (Glowing Spheres)
    const electronCount = 18;
    const sphereGeo = new THREE.SphereGeometry(0.085, 16, 16);

    for (let i = 0; i < electronCount; i++) {
      const electronMat = new THREE.MeshBasicMaterial({
        color: 0x00f5a0
      });
      const electronMesh = new THREE.Mesh(sphereGeo, electronMat);

      // Track relative position t along the curve [0, 1]
      electronParticles.push({
        mesh: electronMesh,
        t: i / electronCount,
        material: electronMat
      });

      wireGroup.add(electronMesh);
    }

    scene.add(wireGroup);
  }

  function updateElectrons(delta) {
    if (!wireCurve || electronParticles.length === 0) return;

    let speed = 0;
    let flowDirection = 0; // +1 or -1
    let electronColor = new THREE.Color(0x00f5a0);

    if (state.isRunning) {
      if (state.mode === 'charge') {
        flowDirection = -1; // Electrons flow toward negative / into cell
        speed = 0.08 + (state.current / 10) * 0.45;
        electronColor.set(0x00d9f5); // Bright Blue/Cyan
      } else if (state.mode === 'discharge') {
        flowDirection = 1; // Electrons flow out of battery
        speed = 0.08 + (state.current / 10) * 0.45;
        electronColor.set(0xffb703); // Energetic Yellow/Amber
      }
    }

    electronParticles.forEach((particle) => {
      // Advance position along curve
      if (speed > 0) {
        particle.t += flowDirection * speed * delta;
        if (particle.t > 1) particle.t -= 1;
        if (particle.t < 0) particle.t += 1;
      }

      // Sample position on curve
      const pos = wireCurve.getPointAt(particle.t);
      particle.mesh.position.copy(pos);

      // Pulse color & size
      particle.material.color.copy(electronColor);
      const scale = state.isRunning ? 1.0 + 0.25 * Math.sin(clock.getElapsedTime() * 10 + particle.t * 6.28) : 0.8;
      particle.mesh.scale.set(scale, scale, scale);
    });
  }

  // =========================================================================
  // 4. BATTERY SIMULATION DYNAMICS & EMISSIVE GLOW
  // =========================================================================
  function updateSimulation(delta) {
    if (state.isRunning) {
      state.timeElapsed += delta;

      // Rate of charge transfer:
      // Delta SoC = (Current * delta) / (CapacityAh * 3600) * 100 * speedUp
      // Let's use an interactive acceleration factor so users don't wait hours to see changes
      const speedUp = 35.0; // accelerated demo factor
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
    // Realistic Li-ion range ~ 3.0V at 0% to 4.2V at 100%
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

    // Update Battery Emissive Glow in Three.js
    updateMaterialsGlow();

    // Update UI Badges & Numbers
    updateTelemetryUI();

    // Update Chart periodically
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
      glowColor.set(0x00d9f5); // Cyan blue
      intensity = state.isRunning ? 0.8 + 0.45 * Math.sin(time * 4) : 0.4;
    } else if (state.mode === 'discharge') {
      glowColor.set(0xff5500); // Warm energetic orange-red
      intensity = state.isRunning ? 0.9 + 0.5 * Math.sin(time * 5) : 0.4;
    } else {
      // Idle mode - calm breathing
      glowColor.set(0x00f5a0);
      intensity = 0.25 + 0.1 * Math.sin(time * 2);
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

    // Dynamic color for SoC bar
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
  // 5. CHART.JS TELEMETRY GRAPH
  // =========================================================================
  function initChart() {
    const ctx = document.getElementById('telemetryChart').getContext('2d');
    const isDark = document.documentElement.getAttribute('data-theme') !== 'light';

    const gridColor = isDark ? 'rgba(255, 255, 255, 0.06)' : 'rgba(0, 0, 0, 0.06)';
    const textColor = isDark ? '#94a3b8' : '#475569';

    // Seed initial historical points
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
          legend: {
            display: false
          },
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

    // Keep sliding window size fixed
    if (telemetryChart.data.labels.length > chartMaxPoints) {
      telemetryChart.data.labels.shift();
      telemetryChart.data.datasets[0].data.shift();
    }

    // Dynamic line color depending on mode
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

    telemetryChart.update('none'); // Update without full layout animation for max performance
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
  // 6. EVENT LISTENERS & USER CONTROLS
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

      // Reset chart data
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
      camera.position.set(4.5, 3.2, 5.8);
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

    // Update Floating Badge
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
  // 7. ANIMATION RENDER LOOP
  // =========================================================================
  function animate() {
    requestAnimationFrame(animate);

    const delta = clock.getDelta();

    // Subtle gentle battery float & rotation when idle or running
    if (batteryGroup) {
      batteryGroup.rotation.y += (state.isRunning ? 0.35 : 0.1) * delta;
    }

    // Controls update
    controls.update();

    // Electron particle motion
    updateElectrons(delta);

    // Physics & Simulation updates
    updateSimulation(delta);

    // Render 3D Frame
    renderer.render(scene, camera);
  }

  // =========================================================================
  // 8. INITIALIZE APPLICATION
  // =========================================================================
  function startApp() {
    initThreeScene();
    initChart();
    setupEventListeners();
    animate();
    console.log('[BATT-SIM 3D] Application initialized successfully.');
  }

  if (document.readyState === 'loading') {
    window.addEventListener('DOMContentLoaded', startApp);
  } else {
    startApp();
  }

  // Expose API on window for interactive programmatic control & testing
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
