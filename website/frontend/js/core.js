import * as THREE from "three";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { ShaderPass } from "three/addons/postprocessing/ShaderPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";

/*
 * TEACHER MODE — "THE NEURAL DIVE," CINEMATIC PASS
 *
 * FOUR PHASES, ONE NUMBER. Approach (0–14%) → Threshold (14–30%) →
 * Freefall (30–86%) → Arrival (86–100%). Same contract as before: every
 * visual system is a pure function of `mainProgress`, computed fresh in
 * `renderForProgress` each frame — no accumulated state, so scrubbing
 * backward always lands on the same frame scrubbing forward would have
 * produced. Arrival is new this pass: it's where the dive actually goes
 * SOMEWHERE — the camera decelerates onto the Vanguard Node waiting at
 * the tunnel's far end, and the page crossfades dark → light around it.
 *
 * WHY THE MONOLITH'S FADE IS DRIVEN BY POSITION, NOT BY PHASE PROGRESS.
 * The chip is a much deeper object now — a 2.6-unit-deep glass shell
 * with five internal glow layers, not a thin flat lid — so tying its
 * opacity to `thresholdT` (an arbitrary 0–1 over a scroll RANGE) would
 * desync from where the camera actually is: at thresholdT=0.3 the
 * camera has only covered 30% of the SCROLL distance, which is not the
 * same as 30% of the way through the physical shell. Instead `crossT`
 * is computed straight from `cameraZ` against the shell's real
 * `SHELL_FRONT_Z`/`SHELL_BACK_Z` bounds — the object fades exactly as
 * the camera's nose is physically inside it, regardless of how the
 * timing curve on top is tuned. `portalFlare` (the die's flash) works
 * the same way: a triangular bump centered on `Math.abs(cameraZ -
 * DIE_Z)`, not on any phase fraction. Position-driven state was already
 * true of the back-face-culling disappearing act from the previous
 * pass (see that reasoning below, extended to five internal layers
 * plus the outer shell — each one is its own convex-ish thin box, so
 * each drops away independently the instant the camera's nose crosses
 * it, reading as "passing through layer after layer" rather than one
 * hard cut).
 *
 * SHARDS: THE SAME GPU-RECYCLE MATH, DELIBERATELY MOVED BACK TO THE
 * CPU. Points and lines (below) recycle entirely inside their vertex
 * shaders because they only ever need a TRANSLATION — mod() arithmetic
 * on one float. Shards need their own tumble ROTATION, independent per
 * instance, which `THREE.InstancedMesh` can only express through a full
 * per-instance 4x4 matrix — and Three.js has no built-in way to compute
 * that matrix on the GPU the way a custom point/line shader can. So
 * `updateShards()` does the identical mod()-based recycle math in plain
 * JS, once per shard, every frame — 350 iterations of vector/quaternion
 * work, the same order of magnitude as the previous pass's ~80-shard
 * facet loop (already proven cheap at 60fps) and nowhere near the
 * 12,000+ count that made a GPU-only approach necessary for the points.
 * Bounded-CPU-loop and GPU-shader-recycle are the same trick at two
 * different scales; which one to reach for depends on whether the
 * per-instance state is just a position (GPU) or a full transform (CPU,
 * as long as the count stays bounded).
 *
 * POST-PROCESSING: HOW A FEW EXTRA SHADER PASSES BUY "EXPENSIVE" FOR
 * CHEAP. `warpPass` (search for it below) is ONE more full-screen
 * `ShaderPass` combining chromatic aberration and a radial motion blur
 * into a single fragment shader. The reason this is the right tool for
 * "make the site feel expensive": its cost is O(screen pixels), a fixed
 * number set once by the canvas resolution — completely independent of
 * how many objects, particles, or triangles are in the scene. Compare
 * that to trying to fake the same look per-object (tinting each
 * material's edges, giving every mesh its own blur shader): that cost
 * is O(objects), grows with the scene, and — critically — never looks
 * consistent, because it's N separate approximations instead of one
 * shared rule applied uniformly to the finished frame. A handful of
 * full-screen passes is how film-grade color grading, chromatic
 * aberration, and vignettes have always been affordable in real-time
 * engines: the whole point of post-processing is that the expensive
 * part (rendering the scene) already happened, and everything after
 * that is comparatively free per additional effect. Two details make
 * this read as "motion," not just "a filter always on": the aberration
 * offset scales with `dist * dist` (distance from screen center), so
 * it's concentrated at the edges exactly where the brief asks for the
 * "rainbow glitch," and both `uAberration` and `uBlur` are driven by
 * `warpBlend` (Freefall's speed, cut back to zero during Arrival) — so
 * it visibly intensifies while falling and relaxes to near-nothing the
 * moment the camera settles on the Node, which is what sells it as
 * sensor overload from velocity rather than a permanent screen filter.
 *
 * Z-FOG: HOW ONE UNIFORM BUYS A SENSE OF INFINITE SCALE. `scene.fog =
 * new THREE.FogExp2(color, density)` mixes every fogged fragment toward
 * `fogColor` by `1 - exp(-density² · distance²)` — squared distance, not
 * linear, which is why it reads as ATMOSPHERE rather than a flat cutoff:
 * nearby objects stay essentially fog-free (the squared term is tiny),
 * then the falloff accelerates hard past a certain distance, so there's
 * no visible "wall" where things disappear, just a graceful dissolve —
 * exactly what real haze/depth-of-field-at-a-distance looks like, and
 * exactly what makes a tunnel with a hard `TUNNEL_LENGTH` boundary read
 * as endless instead of a room with walls. The chip, the shards, and the
 * Vanguard Node get this AUTOMATICALLY — `MeshPhysicalMaterial` and
 * `MeshBasicMaterial` both read `scene.fog` for free (default `fog:
 * true`), no extra code. `dataPoints` and `neuralLines` do NOT: a custom
 * `ShaderMaterial` never gets the built-in fog chunk, so they fake the
 * same falloff manually (`fadeFar` in their fragment shaders) — and it
 * has to be a DIFFERENT technique, not just a manual copy of the same
 * formula, because these two are additively blended. Real fog works by
 * MIXING a fragment's color toward `fogColor` — correct for normal
 * alpha-blended surfaces, where mixing toward the background color is
 * literally what "fading into the haze" means. Additive blending has no
 * "toward" — it only ever ADDS light — so mixing an additive particle's
 * color toward a bright `fogColor` would make it BRIGHTER as it faded,
 * the opposite of the intended effect. The correct additive analogue is
 * fading the contributed brightness/alpha toward zero instead, which is
 * exactly what `fadeFar` already does. One consequence worth noticing:
 * the Vanguard Node, sitting at a fixed `NODE_Z` far down the tunnel,
 * needs no scripted "reveal" animation to grow out of the darkness —
 * real fog already hides it at long range and lets it through as the
 * camera's distance to it shrinks, for free. The manual `arrivalT` ramp
 * on its own opacity/scale/color only adds the FINAL dramatic swallow;
 * the initial "something is glowing out there" is fog doing its job.
 *
 * WHY THE CHIP DISAPPEARS INSTEAD OF BEING HIDDEN (carried over): none
 * of `obsidianMaterial`/`glassMaterial`/the trace/die materials set
 * `side: THREE.DoubleSide`, so only faces whose outward normal points
 * toward the camera rasterize (WebGL's default back-face cull). Once
 * the camera is inside a convex box, every face's outward normal points
 * away from it, so the GPU discards the whole mesh with zero extra
 * logic — the `chipGroup.visible = false` toggle is a pure perf
 * cleanup on top of that, not what actually hides it.
 *
 * "INSTANCED POINTS" (carried over): `THREE.Points` + `BufferGeometry`
 * is the point-cloud equivalent of instancing — one geometry, one draw
 * call, per-vertex attributes (`aSeed`) standing in for per-instance
 * variation, the same contract `InstancedMesh` offers through the
 * point-primitive pipeline instead of the triangle one. The tunnel
 * recycles entirely inside the vertex shader via `mod()` — `position.z`
 * for all 12,000+ points is written ONCE at startup and never touched
 * from JS again; only three scalars (`uCameraZ`, `uSpeed`, `uTime`)
 * change per frame. `frustumCulled = false` on `dataPoints` and
 * `neuralLines` is required, not optional — Three's automatic culling
 * bounds come from the RAW attribute data, which has nothing to do with
 * where the shader actually places those vertices in world space.
 */

gsap.registerPlugin(ScrollTrigger);

(function () {
  "use strict";

  var canvas = document.querySelector(".hero__core");
  if (!canvas) return;

  var prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  var renderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: false });
  } catch (e) {
    // No WebGL available — the canvas stays empty/transparent. Nothing
    // else on the page depends on this succeeding.
    return;
  }

  var pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  renderer.setPixelRatio(pixelRatio);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.15;
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  var scene = new THREE.Scene();

  var VANGUARD_TEAL = new THREE.Color(0x00f0c8);
  var VANGUARD_BLUE = new THREE.Color(0x2f7bff);
  var VANGUARD_ICE = new THREE.Color(0xdafcff);
  var WHITE_COLOR = new THREE.Color(0xffffff);
  var BLACK = new THREE.Color(0x000000);
  var VOID_COLOR = new THREE.Color(0x03040a);
  var ARRIVAL_LIGHT_COLOR = new THREE.Color(0xeef4fb);

  var bgColorScratch = VOID_COLOR.clone();
  scene.background = bgColorScratch;

  var FOG_DENSITY = 0.019;
  scene.fog = new THREE.FogExp2(VOID_COLOR.getHex(), FOG_DENSITY);

  var CAMERA_REST_Z = 6;
  var BASE_FOV = 32;
  var WARP_FOV = 68;
  var SETTLE_FOV = 40;
  var camera = new THREE.PerspectiveCamera(BASE_FOV, 1, 0.1, 300);
  camera.position.set(0, 0, CAMERA_REST_Z);

  var scrollRunway = document.querySelector(".hero__scroll-runway");

  // --- Selective bloom: two full EffectComposers — see Teacher Mode
  // above. BLOOM_LAYER marks which objects glow. ---
  var BLOOM_LAYER = 1;
  var bloomLayer = new THREE.Layers();
  bloomLayer.set(BLOOM_LAYER);

  var darkMaterial = new THREE.MeshBasicMaterial({ color: 0x000000 });
  var materialCache = {};
  function darkenNonBloomed(obj) {
    var isRenderable = obj.isMesh || obj.isPoints || obj.isLineSegments;
    if (isRenderable && bloomLayer.test(obj.layers) === false) {
      materialCache[obj.uuid] = obj.material;
      obj.material = darkMaterial;
    }
  }
  function restoreMaterial(obj) {
    if (materialCache[obj.uuid]) {
      obj.material = materialCache[obj.uuid];
      delete materialCache[obj.uuid];
    }
  }

  var bloomPass = new UnrealBloomPass(new THREE.Vector2(1, 1), 1.4, 0.6, 0.1);
  var bloomComposer = new EffectComposer(renderer);
  bloomComposer.renderToScreen = false;
  bloomComposer.addPass(new RenderPass(scene, camera));
  bloomComposer.addPass(bloomPass);

  var mixPass = new ShaderPass(
    new THREE.ShaderMaterial({
      uniforms: {
        baseTexture: { value: null },
        bloomTexture: { value: bloomComposer.renderTarget2.texture },
      },
      vertexShader:
        "varying vec2 vUv;\nvoid main() {\n  vUv = uv;\n  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);\n}",
      fragmentShader:
        "uniform sampler2D baseTexture;\nuniform sampler2D bloomTexture;\nvarying vec2 vUv;\nvoid main() {\n  gl_FragColor = texture2D(baseTexture, vUv) + texture2D(bloomTexture, vUv);\n}",
    }),
    "baseTexture"
  );

  // --- "The Lusion Shader": chromatic aberration + radial motion blur,
  // combined into ONE full-screen pass. See Teacher Mode,
  // "POST-PROCESSING." Sits after the bloom combine and before
  // OutputPass, so it warps the FINAL, already-bloomed frame. ---
  var warpMaterial = new THREE.ShaderMaterial({
    uniforms: {
      tDiffuse: { value: null },
      uAberration: { value: 0 },
      uBlur: { value: 0 },
      uCenter: { value: new THREE.Vector2(0.5, 0.5) },
    },
    vertexShader:
      "varying vec2 vUv;\nvoid main() {\n  vUv = uv;\n  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);\n}",
    fragmentShader: [
      "uniform sampler2D tDiffuse;",
      "uniform float uAberration;",
      "uniform float uBlur;",
      "uniform vec2 uCenter;",
      "varying vec2 vUv;",
      "void main() {",
      "  vec2 dir = vUv - uCenter;",
      "  float dist = length(dir);",
      "  vec2 ndir = dist > 0.0001 ? dir / dist : vec2(0.0);",
      "  float ca = uAberration * dist * dist;",
      "  float r = texture2D(tDiffuse, vUv - ndir * ca).r;",
      "  float g = texture2D(tDiffuse, vUv).g;",
      "  float b = texture2D(tDiffuse, vUv + ndir * ca).b;",
      "  vec3 color = vec3(r, g, b);",
      "  float total = 1.0;",
      "  for (int i = 1; i <= 5; i++) {",
      "    float t = float(i) / 5.0;",
      "    vec2 offset = ndir * dist * uBlur * t * 0.5;",
      "    color += texture2D(tDiffuse, vUv - offset).rgb;",
      "    total += 1.0;",
      "  }",
      "  color /= total;",
      "  gl_FragColor = vec4(color, 1.0);",
      "}",
    ].join("\n"),
  });
  var warpPass = new ShaderPass(warpMaterial);

  var finalComposer = new EffectComposer(renderer);
  finalComposer.addPass(new RenderPass(scene, camera));
  finalComposer.addPass(mixPass);
  finalComposer.addPass(warpPass);
  finalComposer.addPass(new OutputPass());

  function renderSelectiveBloom() {
    var realBackground = scene.background;
    scene.background = BLACK;
    scene.traverse(darkenNonBloomed);
    bloomComposer.render();
    scene.traverse(restoreMaterial);
    scene.background = realBackground;
    finalComposer.render();
  }

  function resize() {
    var width = window.innerWidth;
    var height = window.innerHeight;
    renderer.setSize(width, height, false);
    bloomComposer.setSize(width, height);
    finalComposer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    updateNodeFillScale();
  }

  function seededRandom(seed) {
    var x = Math.sin(seed * 12.9898) * 43758.5453;
    return x - Math.floor(x);
  }

  function smoothstep(t) {
    t = Math.min(1, Math.max(0, t));
    return t * t * (3 - 2 * t);
  }

  function clamp01(v) {
    return Math.min(1, Math.max(0, v));
  }

  function lerp(a, b, t) {
    return a + (b - a) * t;
  }

  function hexToRgb(hex) {
    var v = parseInt(hex.replace("#", ""), 16);
    return { r: (v >> 16) & 255, g: (v >> 8) & 255, b: v & 255 };
  }
  function lerpColorString(hexA, hexB, t) {
    var a = hexToRgb(hexA);
    var b = hexToRgb(hexB);
    var r = Math.round(a.r + (b.r - a.r) * t);
    var g = Math.round(a.g + (b.g - a.g) * t);
    var bl = Math.round(a.b + (b.b - a.b) * t);
    return "rgb(" + r + "," + g + "," + bl + ")";
  }

  // --- Procedural environment map, recolored to the vibrant teal/blue
  // "Safe Space" palette so the glass shell's reflections carry it. ---
  function buildEnvironmentMap() {
    var envScene = new THREE.Scene();

    var wallMaterial = new THREE.MeshBasicMaterial({ color: 0x050912, side: THREE.BackSide });
    envScene.add(new THREE.Mesh(new THREE.SphereGeometry(6, 16, 16), wallMaterial));

    var tealLight = new THREE.PointLight(0x1fe8c8, 10, 12);
    tealLight.position.set(3, 2, 3);
    envScene.add(tealLight);

    var blueLight = new THREE.PointLight(0x3d6bff, 8, 12);
    blueLight.position.set(-3, -2, -3);
    envScene.add(blueLight);

    var rimLight = new THREE.PointLight(0xdafcff, 6, 12);
    rimLight.position.set(-2, 4, -1);
    envScene.add(rimLight);

    var pmrem = new THREE.PMREMGenerator(renderer);
    var renderTarget = pmrem.fromScene(envScene, 0.04);
    pmrem.dispose();
    return renderTarget.texture;
  }

  scene.environment = buildEnvironmentMap();

  var ambient = new THREE.AmbientLight(0x1a2440, 0.4);
  scene.add(ambient);

  var keyLight = new THREE.PointLight(0x8ff5ff, 4, 20);
  keyLight.position.set(2, 2, 4);
  scene.add(keyLight);

  var rimLight = new THREE.PointLight(0x3d6bff, 3, 20);
  rimLight.position.set(-3, -1.5, 2);
  scene.add(rimLight);

  function frustumHalfSizeAtDistance(fovDeg, aspect, distance) {
    var heightHalf = Math.tan(THREE.MathUtils.degToRad(fovDeg / 2)) * distance;
    return { halfWidth: heightHalf * aspect, halfHeight: heightHalf };
  }

  // =========================================================================
  // ACT 1 — THE VANGUARD ENGINE, NOW A MONOLITH. A deep (2.6-unit) glass
  // shell fixed at the origin, an obsidian cap facing the camera, an
  // obsidian backing deep inside, five internal glowing circuit-trace
  // layers stacked along the dive axis, and the die itself at the back —
  // the thing the camera is actually diving toward. Everything here is
  // still fixed in world space; only the camera moves (see Teacher Mode).
  // =========================================================================
  var tiltGroup = new THREE.Group(); // owned by mouse-tilt, faded out by the dive
  scene.add(tiltGroup);

  var chipGroup = new THREE.Group();
  tiltGroup.add(chipGroup);

  var SHELL_FRONT_Z = 1.3;
  var SHELL_BACK_Z = -1.3;
  var DIE_Z = -1.12;
  var PORTAL_FLARE_WIDTH = 0.6;

  var obsidianMaterial = new THREE.MeshPhysicalMaterial({
    color: 0x030305,
    metalness: 0.25,
    roughness: 0.3,
    clearcoat: 1,
    clearcoatRoughness: 0.2,
    envMapIntensity: 1.1,
    transparent: true,
    opacity: 1,
  });
  var obsidianCap = new THREE.Mesh(new THREE.BoxGeometry(2.05, 2.05, 0.26), obsidianMaterial);
  obsidianCap.position.z = 1.17;
  chipGroup.add(obsidianCap);

  var obsidianBacking = new THREE.Mesh(new THREE.BoxGeometry(2.05, 2.05, 0.16), obsidianMaterial);
  obsidianBacking.position.z = -1.22;
  chipGroup.add(obsidianBacking);

  // ior 1.45 (architectural glass, not diamond), a dark attenuationColor
  // pulled toward VANGUARD_TEAL so light transmitted THROUGH the shell
  // picks up a tinted "internal glow" quality rather than clear window
  // glass. thickness: 1.4 gives the transmission real depth to bend
  // through given how deep this shell now is.
  var glassMaterial = new THREE.MeshPhysicalMaterial({
    color: 0x060a10,
    metalness: 0,
    roughness: 0.035,
    transmission: 1.0,
    thickness: 1.4,
    ior: 1.45,
    attenuationColor: VANGUARD_TEAL.clone(),
    attenuationDistance: 0.9,
    clearcoat: 1,
    clearcoatRoughness: 0.03,
    envMapIntensity: 1.7,
    transparent: true,
    opacity: 1,
  });
  var shell = new THREE.Mesh(new THREE.BoxGeometry(2.0, 2.0, 2.6), glassMaterial);
  chipGroup.add(shell);

  var DIE_COOL = VANGUARD_TEAL.clone();
  var DIE_HOT = new THREE.Color(0xffffff).lerp(VANGUARD_ICE, 0.3);
  var dieMaterial = new THREE.MeshBasicMaterial({ color: DIE_COOL, transparent: true, opacity: 1 });
  var dieMesh = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.5, 0.1), dieMaterial);
  dieMesh.position.z = DIE_Z;
  dieMesh.layers.enable(BLOOM_LAYER);
  chipGroup.add(dieMesh);

  var dieLight = new THREE.PointLight(VANGUARD_TEAL.getHex(), 1, 4.5);
  dieLight.position.z = DIE_Z;
  chipGroup.add(dieLight);

  // Portal rings: flat, facing the camera by default (TorusGeometry lies
  // in the XY plane) — brighten via `portalFlare` (a position-based bump
  // around DIE_Z, see Teacher Mode) as the camera passes the die.
  var portalMaterial1 = new THREE.MeshBasicMaterial({
    color: VANGUARD_TEAL, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false,
  });
  var portalMaterial2 = new THREE.MeshBasicMaterial({
    color: VANGUARD_BLUE, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false,
  });
  var portalRing1 = new THREE.Mesh(new THREE.TorusGeometry(0.34, 0.007, 8, 48), portalMaterial1);
  var portalRing2 = new THREE.Mesh(new THREE.TorusGeometry(0.47, 0.006, 8, 48), portalMaterial2);
  portalRing1.position.z = DIE_Z + 0.06;
  portalRing2.position.z = DIE_Z + 0.06;
  portalRing1.layers.enable(BLOOM_LAYER);
  portalRing2.layers.enable(BLOOM_LAYER);
  chipGroup.add(portalRing1, portalRing2);

  // --- Circuit traces: one InstancedMesh, one draw call, instances
  // distributed across FIVE Z depths inside the shell — "visible
  // internal layers" made of the same circuit-trace geometry, alternating
  // teal/blue/ice per layer via per-instance color (InstancedMesh's
  // native `setColorAt`, no extra draw calls). Deliberately thin/gappy
  // rather than solid plates, so the camera can see toward the die
  // through them during Approach instead of them blocking the view. ---
  function buildRadialTraces(count, innerR, outerR, seedOffset) {
    var out = [];
    for (var i = 0; i < count; i++) {
      var angle = (i / count) * Math.PI * 2 + seededRandom(i * 3.7 + seedOffset) * 0.2;
      var start = innerR + seededRandom(i * 1.9 + seedOffset + 2) * 0.06;
      var length = Math.max(outerR - start - seededRandom(i * 4.4 + seedOffset + 1) * 0.25, 0.12);
      out.push({ cx: Math.cos(angle) * (start + length / 2), cy: Math.sin(angle) * (start + length / 2), dir: angle, length: length });
    }
    return out;
  }
  function buildRingTraces(count, radius, seedOffset) {
    var out = [];
    for (var i = 0; i < count; i++) {
      var angle = (i / count) * Math.PI * 2 + seededRandom(i * 2.2 + seedOffset) * 0.05;
      if (seededRandom(i * 5.1 + seedOffset) < 0.3) continue; // gappy, PCB-like, not a solid ring
      var arc = ((Math.PI * 2) / count) * 0.7;
      out.push({ cx: Math.cos(angle) * radius, cy: Math.sin(angle) * radius, dir: angle + Math.PI / 2, length: radius * arc });
    }
    return out;
  }
  function buildTraceLayer(zDepth, colorHex, seedBase) {
    var radial = buildRadialTraces(15, 0.36, 0.92, seedBase);
    var ring = buildRingTraces(11, 0.66, seedBase + 31);
    var combined = radial.concat(ring);
    combined.forEach(function (t) {
      t.z = zDepth;
      t.color = colorHex;
    });
    return combined;
  }

  var TRACE_LAYERS = [
    { z: 0.82, hex: 0x00f0c8, seed: 3 },
    { z: 0.38, hex: 0x2f7bff, seed: 47 },
    { z: -0.06, hex: 0x00f0c8, seed: 91 },
    { z: -0.52, hex: 0x2f7bff, seed: 137 },
    { z: -0.98, hex: 0xdafcff, seed: 181 },
  ];
  var traceData = [];
  TRACE_LAYERS.forEach(function (layer) {
    traceData = traceData.concat(buildTraceLayer(layer.z, layer.hex, layer.seed));
  });
  var TRACE_COUNT = traceData.length;

  var traceMaterial = new THREE.MeshBasicMaterial({
    vertexColors: true, transparent: true, opacity: 0.16, blending: THREE.AdditiveBlending, depthWrite: false,
  });
  var traceMesh = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), traceMaterial, TRACE_COUNT);
  traceMesh.layers.enable(BLOOM_LAYER);
  chipGroup.add(traceMesh);

  var traceMatrix = new THREE.Matrix4();
  var traceQuat = new THREE.Quaternion();
  var traceScale = new THREE.Vector3();
  var traceZAxis = new THREE.Vector3(0, 0, 1);
  var traceColorScratch = new THREE.Color();
  traceData.forEach(function (t, i) {
    traceQuat.setFromAxisAngle(traceZAxis, t.dir);
    traceScale.set(t.length, 0.014, 0.008);
    traceMatrix.compose(new THREE.Vector3(t.cx, t.cy, t.z), traceQuat, traceScale);
    traceMesh.setMatrixAt(i, traceMatrix);
    traceColorScratch.set(t.color).multiplyScalar(0.85);
    traceMesh.setColorAt(i, traceColorScratch);
  });
  traceMesh.instanceMatrix.needsUpdate = true;
  traceMesh.instanceColor.needsUpdate = true;

  // --- Data rays: same InstancedMesh technique, oriented outward in
  // full 3D from the die — "rays of data ... emitting from the chip." ---
  var RAY_COUNT = 30;
  var rayData = [];
  for (var ri = 0; ri < RAY_COUNT; ri++) {
    var theta = seededRandom(ri * 2.3 + 5) * Math.PI * 2;
    var phi = Math.acos(2 * seededRandom(ri * 3.1 + 8) - 1);
    var rayDir = new THREE.Vector3(Math.sin(phi) * Math.cos(theta), Math.sin(phi) * Math.sin(theta), Math.cos(phi) * 0.7).normalize();
    rayData.push({ dir: rayDir, length: 1.3 + seededRandom(ri * 6.6 + 2) * 3.4 });
  }
  var rayMaterial = new THREE.MeshBasicMaterial({
    color: VANGUARD_ICE, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false,
  });
  var rayMesh = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), rayMaterial, RAY_COUNT);
  rayMesh.layers.enable(BLOOM_LAYER);
  chipGroup.add(rayMesh);

  var rayMatrix = new THREE.Matrix4();
  var rayQuat = new THREE.Quaternion();
  var rayScale = new THREE.Vector3();
  var rayXAxis = new THREE.Vector3(1, 0, 0);
  var dieOrigin = new THREE.Vector3(0, 0, DIE_Z);
  rayData.forEach(function (r, i) {
    rayQuat.setFromUnitVectors(rayXAxis, r.dir);
    rayScale.set(r.length, 0.01, 0.01);
    rayMatrix.compose(
      dieOrigin.clone().addScaledVector(r.dir, r.length / 2 + 0.18),
      rayQuat,
      rayScale
    );
    rayMesh.setMatrixAt(i, rayMatrix);
  });
  rayMesh.instanceMatrix.needsUpdate = true;

  // Breathing: a slow scale loop on the whole monolith — skipped under
  // reduced motion.
  if (!prefersReducedMotion) {
    gsap.to(chipGroup.scale, {
      x: 1.025, y: 1.025, z: 1.025, duration: 3.4, ease: "sine.inOut", yoyo: true, repeat: -1,
    });
  }

  // =========================================================================
  // ACT 2 — THE VOLUMETRIC TUNNEL. Three systems sharing one recycling
  // scheme: 12,000+ points (fine dust), 1,800 line segments (streak
  // lines), and 350 instanced glass shards (the bigger debris the brief
  // asks for). Points/lines recycle on the GPU (see Teacher Mode); shards
  // recycle in a bounded CPU loop because they need individual rotation.
  // None of these three are parented to `tiltGroup` — the tunnel is a
  // fixed world-space structure the camera flies through.
  // =========================================================================
  var TUNNEL_LENGTH = 220;
  var PARTICLE_COUNT = 12000;
  var LINE_COUNT = 1800;
  var SHARD_COUNT = 350;
  var TUNNEL_RADIUS_MIN = 0.6;
  var TUNNEL_RADIUS_MAX = 9.5;
  var AUTO_DRIFT_SPEED = 5.5;

  function buildTunnelPoint(seed, radiusMin, radiusMax, biasPow) {
    var angle = seededRandom(seed * 3.11 + 1.7) * Math.PI * 2;
    var r = radiusMin + (radiusMax - radiusMin) * Math.pow(seededRandom(seed * 7.31 + 4.2), biasPow);
    var z = seededRandom(seed * 2.03 + 9.9) * TUNNEL_LENGTH;
    return { x: Math.cos(angle) * r, y: Math.sin(angle) * r, z: z, seed: seededRandom(seed * 5.55 + 0.31) };
  }

  // Copied by reference (not deep-cloned) into both materials below via
  // Object.assign, so each of these stays ONE object that both
  // pointsMaterial.uniforms and lineMaterial.uniforms point to — mutating
  // sharedUniforms.uCameraZ.value once in renderForProgress updates what
  // both shaders read, with no per-material bookkeeping.
  var sharedUniforms = {
    uCameraZ: { value: CAMERA_REST_Z },
    uTime: { value: 0 },
    uSpeed: { value: 0 },
    uDiveT: { value: 0 },
    uTunnelLength: { value: TUNNEL_LENGTH },
    uAutoDrift: { value: AUTO_DRIFT_SPEED },
    uColorA: { value: VANGUARD_TEAL.clone() },
    uColorB: { value: VANGUARD_BLUE.clone() },
  };

  var pointsPositions = new Float32Array(PARTICLE_COUNT * 3);
  var pointsSeeds = new Float32Array(PARTICLE_COUNT);
  for (var pi = 0; pi < PARTICLE_COUNT; pi++) {
    var p = buildTunnelPoint(pi, TUNNEL_RADIUS_MIN, TUNNEL_RADIUS_MAX, 0.5);
    pointsPositions[pi * 3] = p.x;
    pointsPositions[pi * 3 + 1] = p.y;
    pointsPositions[pi * 3 + 2] = p.z;
    pointsSeeds[pi] = p.seed;
  }
  var pointsGeometry = new THREE.BufferGeometry();
  pointsGeometry.setAttribute("position", new THREE.BufferAttribute(pointsPositions, 3));
  pointsGeometry.setAttribute("aSeed", new THREE.BufferAttribute(pointsSeeds, 1));

  var pointsMaterial = new THREE.ShaderMaterial({
    uniforms: Object.assign({}, sharedUniforms, {
      uPixelRatio: { value: pixelRatio },
      uBaseSize: { value: 10.0 },
    }),
    vertexShader: [
      "attribute float aSeed;",
      "uniform float uCameraZ;",
      "uniform float uTime;",
      "uniform float uSpeed;",
      "uniform float uTunnelLength;",
      "uniform float uAutoDrift;",
      "uniform float uPixelRatio;",
      "uniform float uBaseSize;",
      "varying float vAlpha;",
      "varying float vSeed;",
      "void main() {",
      "  float half = uTunnelLength * 0.5;",
      "  float driftZ = uCameraZ - uTime * uAutoDrift * uSpeed;",
      "  float rel = mod(position.z - driftZ, uTunnelLength) - half;",
      "  vec3 worldPos = vec3(position.x, position.y, uCameraZ + rel);",
      "  vec4 mvPosition = modelViewMatrix * vec4(worldPos, 1.0);",
      "  gl_Position = projectionMatrix * mvPosition;",
      "  float dist = max(-mvPosition.z, 0.001);",
      "  float fadeNear = smoothstep(0.15, 2.5, dist);",
      "  float fadeFar = 1.0 - smoothstep(50.0, 88.0, dist);",
      "  vAlpha = fadeNear * fadeFar;",
      "  vSeed = aSeed;",
      "  float sizeBoost = 1.0 + uSpeed * 2.2;",
      "  float size = (uBaseSize + aSeed * 6.0) * sizeBoost * uPixelRatio * (110.0 / dist);",
      "  gl_PointSize = min(size, 64.0);",
      "}",
    ].join("\n"),
    fragmentShader: [
      "uniform vec3 uColorA;",
      "uniform vec3 uColorB;",
      "uniform float uDiveT;",
      "varying float vAlpha;",
      "varying float vSeed;",
      "void main() {",
      "  vec2 c = gl_PointCoord - 0.5;",
      "  float d = length(c);",
      "  if (d > 0.5) discard;",
      "  float glow = smoothstep(0.5, 0.0, d);",
      "  float brightness = 0.14 + uDiveT * 0.86;",
      "  vec3 color = mix(uColorA, uColorB, vSeed) * brightness;",
      "  gl_FragColor = vec4(color, glow * vAlpha);",
      "}",
    ].join("\n"),
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });

  var dataPoints = new THREE.Points(pointsGeometry, pointsMaterial);
  dataPoints.frustumCulled = false; // see Teacher Mode — raw attribute bounds don't match shader-recycled world positions
  dataPoints.layers.enable(BLOOM_LAYER);
  scene.add(dataPoints);

  var linePositions = new Float32Array(LINE_COUNT * 2 * 3);
  var lineEnds = new Float32Array(LINE_COUNT * 2);
  var lineSeeds = new Float32Array(LINE_COUNT * 2);
  for (var li = 0; li < LINE_COUNT; li++) {
    var lp = buildTunnelPoint(li + 90000, TUNNEL_RADIUS_MIN, TUNNEL_RADIUS_MAX * 1.05, 0.65);
    for (var v = 0; v < 2; v++) {
      var idx = li * 2 + v;
      linePositions[idx * 3] = lp.x;
      linePositions[idx * 3 + 1] = lp.y;
      linePositions[idx * 3 + 2] = lp.z;
      lineEnds[idx] = v;
      lineSeeds[idx] = lp.seed;
    }
  }
  var lineGeometry = new THREE.BufferGeometry();
  lineGeometry.setAttribute("position", new THREE.BufferAttribute(linePositions, 3));
  lineGeometry.setAttribute("aEnd", new THREE.BufferAttribute(lineEnds, 1));
  lineGeometry.setAttribute("aSeed", new THREE.BufferAttribute(lineSeeds, 1));

  var lineMaterial = new THREE.ShaderMaterial({
    uniforms: Object.assign({}, sharedUniforms, {
      uStreakBase: { value: 0.15 },
      uStreakRange: { value: 5.5 },
    }),
    vertexShader: [
      "attribute float aEnd;",
      "attribute float aSeed;",
      "uniform float uCameraZ;",
      "uniform float uTime;",
      "uniform float uSpeed;",
      "uniform float uTunnelLength;",
      "uniform float uAutoDrift;",
      "uniform float uStreakBase;",
      "uniform float uStreakRange;",
      "varying float vAlpha;",
      "varying float vSeed;",
      "void main() {",
      "  float half = uTunnelLength * 0.5;",
      "  float driftZ = uCameraZ - uTime * uAutoDrift * uSpeed;",
      "  float rel = mod(position.z - driftZ, uTunnelLength) - half;",
      "  float streak = uStreakBase + uStreakRange * uSpeed;",
      "  float z = uCameraZ + rel - aEnd * streak;",
      "  vec3 worldPos = vec3(position.x, position.y, z);",
      "  vec4 mvPosition = modelViewMatrix * vec4(worldPos, 1.0);",
      "  gl_Position = projectionMatrix * mvPosition;",
      "  float dist = max(-mvPosition.z, 0.001);",
      "  float fadeNear = smoothstep(0.15, 2.5, dist);",
      "  float fadeFar = 1.0 - smoothstep(50.0, 88.0, dist);",
      "  vAlpha = fadeNear * fadeFar * (1.0 - aEnd * 0.65);",
      "  vSeed = aSeed;",
      "}",
    ].join("\n"),
    fragmentShader: [
      "uniform vec3 uColorA;",
      "uniform vec3 uColorB;",
      "uniform float uDiveT;",
      "varying float vAlpha;",
      "varying float vSeed;",
      "void main() {",
      "  float brightness = 0.1 + uDiveT * 0.7;",
      "  vec3 color = mix(uColorA, uColorB, vSeed) * brightness;",
      "  gl_FragColor = vec4(color, vAlpha * brightness);",
      "}",
    ].join("\n"),
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });

  var neuralLines = new THREE.LineSegments(lineGeometry, lineMaterial);
  neuralLines.frustumCulled = false; // same reason as dataPoints, see Teacher Mode
  neuralLines.layers.enable(BLOOM_LAYER);
  scene.add(neuralLines);

  // --- Glass shards: InstancedMesh, CPU-recycled — see Teacher Mode,
  // "SHARDS." Tetrahedron geometry (cheap, angular, catches bloom well),
  // per-instance teal/blue tint via instanceColor, real `scene.fog`
  // (MeshBasicMaterial supports it automatically). ---
  function buildShardData(count) {
    var out = [];
    for (var i = 0; i < count; i++) {
      var angle = seededRandom(i * 4.13 + 21) * Math.PI * 2;
      var r = 0.5 + 6.0 * Math.pow(seededRandom(i * 8.7 + 3), 0.6);
      var axis = new THREE.Vector3(
        seededRandom(i * 3 + 1) - 0.5,
        seededRandom(i * 3 + 2) - 0.5,
        seededRandom(i * 3 + 3) - 0.5
      ).normalize();
      out.push({
        x: Math.cos(angle) * r,
        y: Math.sin(angle) * r,
        baseZ: seededRandom(i * 1.77 + 55) * TUNNEL_LENGTH,
        colorMix: seededRandom(i * 9.2 + 7),
        tumbleAxis: axis,
        tumbleSpeed: 0.4 + seededRandom(i * 5 + 9) * 1.6,
        phase: seededRandom(i * 6 + 13) * Math.PI * 2,
        scale: 0.6 + seededRandom(i * 2 + 4) * 0.9,
      });
    }
    return out;
  }
  var shardData = buildShardData(SHARD_COUNT);

  var shardMaterial = new THREE.MeshBasicMaterial({
    vertexColors: true, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false, fog: true,
  });
  var shardMesh = new THREE.InstancedMesh(new THREE.TetrahedronGeometry(0.085), shardMaterial, SHARD_COUNT);
  shardMesh.frustumCulled = false; // matrices are recomputed every frame from world-space positions; see dataPoints note
  scene.add(shardMesh);

  var shardColorScratch = new THREE.Color();
  shardData.forEach(function (s, i) {
    shardColorScratch.copy(VANGUARD_TEAL).lerp(VANGUARD_ICE, s.colorMix);
    shardMesh.setColorAt(i, shardColorScratch);
  });
  shardMesh.instanceColor.needsUpdate = true;

  var shardMatrixScratch = new THREE.Matrix4();
  var shardQuatScratch = new THREE.Quaternion();
  var shardPosScratch = new THREE.Vector3();
  var shardScaleScratch = new THREE.Vector3();

  function updateShards() {
    var t = sharedUniforms.uTime.value;
    var half = TUNNEL_LENGTH * 0.5;
    var driftZ = lastCameraZ - t * AUTO_DRIFT_SPEED * lastFreefallT;
    for (var i = 0; i < SHARD_COUNT; i++) {
      var s = shardData[i];
      var rel = (((s.baseZ - driftZ) % TUNNEL_LENGTH) + TUNNEL_LENGTH) % TUNNEL_LENGTH - half;
      shardPosScratch.set(s.x, s.y, lastCameraZ + rel);
      shardQuatScratch.setFromAxisAngle(s.tumbleAxis, t * s.tumbleSpeed + s.phase);
      var scale = s.scale * (0.85 + 0.15 * Math.sin(t * 1.3 + s.phase));
      shardScaleScratch.setScalar(scale);
      shardMatrixScratch.compose(shardPosScratch, shardQuatScratch, shardScaleScratch);
      shardMesh.setMatrixAt(i, shardMatrixScratch);
    }
    shardMesh.instanceMatrix.needsUpdate = true;
  }

  // =========================================================================
  // ACT 3 — THE DESTINATION. A wireframe Vanguard Node sitting at a fixed
  // depth far down the tunnel. It needs no scripted "reveal" — real
  // `scene.fog` already hides it at range and lets it through as the
  // camera's distance shrinks (see Teacher Mode, "Z-FOG"). `arrivalT`
  // only drives the final dramatic swallow: scale to fill the frame,
  // color crossfading to white, and the page's own theme going with it.
  // =========================================================================
  var NODE_Z = -78;
  var NODE_GLOW_BASE_RADIUS = 0.85;

  var nodeGroup = new THREE.Group();
  nodeGroup.position.z = NODE_Z;
  scene.add(nodeGroup);

  var nodeWireMaterial = new THREE.MeshBasicMaterial({
    color: VANGUARD_TEAL, wireframe: true, transparent: true, opacity: 0.75, fog: true,
  });
  var nodeWire = new THREE.Mesh(new THREE.IcosahedronGeometry(0.6, 1), nodeWireMaterial);
  nodeWire.layers.enable(BLOOM_LAYER);
  nodeGroup.add(nodeWire);

  var nodeGlowMaterial = new THREE.MeshBasicMaterial({
    color: VANGUARD_BLUE, transparent: true, opacity: 0.22, blending: THREE.AdditiveBlending, depthWrite: false, fog: true,
  });
  var nodeGlow = new THREE.Mesh(new THREE.IcosahedronGeometry(NODE_GLOW_BASE_RADIUS, 1), nodeGlowMaterial);
  nodeGlow.layers.enable(BLOOM_LAYER);
  nodeGroup.add(nodeGlow);

  var nodeLight = new THREE.PointLight(VANGUARD_TEAL.getHex(), 2, 40);
  nodeGroup.add(nodeLight);

  var nodeFillScale = 1;
  function updateNodeFillScale() {
    var distance = CAMERA_ARRIVAL_END_Z - NODE_Z;
    var frustum = frustumHalfSizeAtDistance(SETTLE_FOV, camera.aspect || 1, distance);
    nodeFillScale = (Math.max(frustum.halfWidth, frustum.halfHeight) * 1.2) / NODE_GLOW_BASE_RADIUS;
  }

  var nodeColorScratch = new THREE.Color();
  var nodeGlowColorScratch = new THREE.Color();

  // --- Timeline boundaries: Approach 0–14%, Threshold 14–30%,
  // Freefall 30–86%, Arrival 86–100%. See Teacher Mode. ---
  var APPROACH_END = 0.14;
  var THRESHOLD_END = 0.3;
  var FREEFALL_END = 0.86;
  var CAMERA_APPROACH_NEAR_Z = 1.9;
  var CAMERA_THRESHOLD_EXIT_Z = -1.9;
  var CAMERA_FREEFALL_END_Z = -64;
  var CAMERA_ARRIVAL_END_Z = -75.4;

  updateNodeFillScale();

  var DARK_BG = "#0d0d0f";
  var LIGHT_BG = "#eef4fb";
  var DARK_TEXT_PRIMARY = "#f5f5f2";
  var LIGHT_TEXT_PRIMARY = "#23262b";
  var DARK_TEXT_MUTED = "#9a9aa1";
  var LIGHT_TEXT_MUTED = "#565c66";

  var root = document.documentElement;
  var dieColorScratch = new THREE.Color();

  var lastApproachT = 0;
  var lastThresholdT = 0;
  var lastFreefallT = 0;
  var lastArrivalT = 0;
  var lastCameraZ = CAMERA_REST_Z;

  // The single per-frame fan-out: one scroll-timeline progress number in,
  // every visual system — camera Z/FOV, monolith fade, portal flare, ray
  // glow, tunnel brightness/speed, the warp shader, the Node, the page's
  // own theme — updated out. Pure function of `progress` (see Teacher Mode).
  function renderForProgress(progress) {
    var approachT = smoothstep(progress / APPROACH_END);
    var thresholdT = smoothstep((progress - APPROACH_END) / (THRESHOLD_END - APPROACH_END));
    var freefallRaw = clamp01((progress - THRESHOLD_END) / (FREEFALL_END - THRESHOLD_END));
    var freefallT = freefallRaw * freefallRaw * freefallRaw; // cubic ease-in
    var arrivalRaw = clamp01((progress - FREEFALL_END) / (1 - FREEFALL_END));
    var arrivalT = smoothstep(arrivalRaw);
    lastApproachT = approachT;
    lastThresholdT = thresholdT;
    lastFreefallT = freefallT;
    lastArrivalT = arrivalT;

    var cameraZ;
    if (progress <= APPROACH_END) {
      cameraZ = lerp(CAMERA_REST_Z, CAMERA_APPROACH_NEAR_Z, approachT);
    } else if (progress <= THRESHOLD_END) {
      cameraZ = lerp(CAMERA_APPROACH_NEAR_Z, CAMERA_THRESHOLD_EXIT_Z, thresholdT);
    } else if (progress <= FREEFALL_END) {
      cameraZ = lerp(CAMERA_THRESHOLD_EXIT_Z, CAMERA_FREEFALL_END_Z, freefallT);
    } else {
      cameraZ = lerp(CAMERA_FREEFALL_END_Z, CAMERA_ARRIVAL_END_Z, arrivalT);
    }
    camera.position.z = cameraZ;
    lastCameraZ = cameraZ;

    camera.fov = progress <= FREEFALL_END ? lerp(BASE_FOV, WARP_FOV, freefallT) : lerp(WARP_FOV, SETTLE_FOV, arrivalT);
    camera.updateProjectionMatrix();

    // Monolith: position-based fade, not phase-based — see Teacher Mode.
    var crossSpan = SHELL_FRONT_Z - SHELL_BACK_Z;
    var crossT = smoothstep((SHELL_FRONT_Z - cameraZ) / crossSpan);
    var chipOpacity = 1 - crossT;
    obsidianMaterial.opacity = chipOpacity;
    glassMaterial.opacity = chipOpacity;
    chipGroup.visible = chipOpacity > 0.015;

    var rayGlow = approachT * (1 - crossT);
    rayMaterial.opacity = rayGlow * 0.8;

    var distToDie = Math.abs(cameraZ - DIE_Z);
    var portalFlare = clamp01(1 - distToDie / PORTAL_FLARE_WIDTH);
    portalMaterial1.opacity = portalFlare * 0.9;
    portalMaterial2.opacity = portalFlare * 0.7;

    var dieHeat = clamp01(approachT * 0.4 + portalFlare * 1.2);
    dieColorScratch.copy(DIE_COOL).lerp(DIE_HOT, dieHeat);
    dieMaterial.color.copy(dieColorScratch);
    dieLight.intensity = 1 + approachT * 3 + portalFlare * 7;
    dieLight.color.copy(dieColorScratch);

    traceMaterial.opacity = 0.16 + rayGlow * 0.5 + portalFlare * 0.4;

    // The Matrix: brightness (uDiveT) ramps in gently starting mid-
    // Approach, reaches full vibrance partway into Freefall, then fades
    // back out across Arrival so the tunnel doesn't linger against the
    // new white backdrop once we've landed on the Node.
    var diveT = smoothstep((progress - 0.06) / 0.6);
    var tunnelFade = diveT * (1 - arrivalT);
    sharedUniforms.uCameraZ.value = cameraZ;
    sharedUniforms.uSpeed.value = freefallT;
    sharedUniforms.uDiveT.value = tunnelFade;
    shardMaterial.opacity = 0.55 * tunnelFade;

    // The Lusion shader: aberration/blur both driven by Freefall speed,
    // cut back to zero across Arrival — see Teacher Mode, "POST-PROCESSING."
    var warpBlend = freefallT * (1 - arrivalT);
    warpMaterial.uniforms.uAberration.value = 0.0015 + warpBlend * 0.018;
    warpMaterial.uniforms.uBlur.value = warpBlend * 0.5;

    // Act 3: the Node grows into the frame and crossfades to white —
    // see Teacher Mode, "Z-FOG" for why it's already visible before this.
    nodeGroup.scale.setScalar(lerp(1, nodeFillScale, arrivalT));
    nodeColorScratch.copy(VANGUARD_TEAL).lerp(WHITE_COLOR, arrivalT);
    nodeWireMaterial.color.copy(nodeColorScratch);
    nodeWireMaterial.opacity = 0.75 + arrivalT * 0.25;
    nodeGlowColorScratch.copy(VANGUARD_BLUE).lerp(WHITE_COLOR, arrivalT);
    nodeGlowMaterial.color.copy(nodeGlowColorScratch);
    nodeGlowMaterial.opacity = 0.22 + arrivalT * 0.55;
    nodeLight.intensity = 2 + arrivalT * 15;
    nodeLight.color.copy(nodeColorScratch);

    // Theme crossfade: the WebGL void, the fog it fades into, and the
    // page's own CSS custom properties all read the same arrivalT, so
    // the 3D scene and the DOM chrome can never drift out of sync.
    bgColorScratch.copy(VOID_COLOR).lerp(ARRIVAL_LIGHT_COLOR, arrivalT);
    scene.fog.color.copy(bgColorScratch);
    root.style.setProperty("--color-bg", lerpColorString(DARK_BG, LIGHT_BG, arrivalT));
    root.style.setProperty("--color-text-primary", lerpColorString(DARK_TEXT_PRIMARY, LIGHT_TEXT_PRIMARY, arrivalT));
    root.style.setProperty("--color-text-muted", lerpColorString(DARK_TEXT_MUTED, LIGHT_TEXT_MUTED, arrivalT));

    // Hero text/chrome: fades out across the Approach so nothing is
    // still asking to be read once the dive begins.
    var diveFade = smoothstep(progress / (APPROACH_END * 0.9));
    root.style.setProperty("--dive-fade", diveFade.toFixed(3));
  }

  // --- Cursor tracking for the monolith's subtle tilt + key-light nudge —
  // same exponential-smoothing technique as js/main.js's grain/glow drift.
  // Its visible effect fades naturally as the monolith itself fades out. ---
  var TILT_SMOOTHING = 0.05;
  var tiltTargetX = 0, tiltTargetY = 0;
  var tiltCurrentX = 0, tiltCurrentY = 0;

  if (!prefersReducedMotion) {
    window.addEventListener("mousemove", function (event) {
      tiltTargetX = (event.clientY / window.innerHeight - 0.5) * 2; // -1 .. 1
      tiltTargetY = (event.clientX / window.innerWidth - 0.5) * 2;  // -1 .. 1
    });
  }

  resize();
  renderForProgress(0);
  updateShards();

  var mainProgress = 0;
  ScrollTrigger.create({
    trigger: scrollRunway,
    start: "top top",
    end: "bottom bottom",
    scrub: prefersReducedMotion ? true : 1, // true = instant tracking, no eased lag, for reduced motion
    onUpdate: function (self) {
      mainProgress = self.progress;
    },
  });

  ScrollTrigger.refresh();
  window.addEventListener("resize", function () {
    resize();
    ScrollTrigger.refresh();
  });

  // --- The render loop, driven by GSAP's own ticker — the breathing
  // tween and ScrollTrigger's scrub both already run on gsap's internal
  // ticker, so rendering from it keeps everything on one shared clock
  // instead of racing rAF loops. ---
  var clockStart = performance.now();
  gsap.ticker.add(function () {
    var now = performance.now();
    if (!prefersReducedMotion) {
      sharedUniforms.uTime.value = (now - clockStart) / 1000;
    }

    renderForProgress(mainProgress);
    updateShards();

    if (!prefersReducedMotion) {
      tiltCurrentX += (tiltTargetX - tiltCurrentX) * TILT_SMOOTHING;
      tiltCurrentY += (tiltTargetY - tiltCurrentY) * TILT_SMOOTHING;
      tiltGroup.rotation.set(tiltCurrentX * 0.12, tiltCurrentY * 0.12, 0);

      keyLight.position.x = 2 + tiltCurrentY * 1.2;
      keyLight.position.y = 2 - tiltCurrentX * 1.2;

      var pulseSpeed = 0.0018 + lastApproachT * 0.0035 + lastThresholdT * 0.005;
      var diePulse = 1 + Math.sin(now * pulseSpeed) * 0.08;
      dieMesh.scale.setScalar(diePulse);
    }

    renderSelectiveBloom();
  });
})();
