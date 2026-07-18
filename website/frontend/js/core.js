import * as THREE from "three";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { ShaderPass } from "three/addons/postprocessing/ShaderPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";

/*
 * TEACHER MODE — read this before the code below.
 *
 * THE SCROLL TIMELINE: ONE SCRUBBED SCROLLTRIGGER, ONE FAN-OUT FUNCTION.
 * There's no idle-corner-then-snap-to-center choreography this time —
 * the brief describes the Shell as centered from the start, so the
 * whole experience is a single `ScrollTrigger.create({ ..., scrub: 1,
 * onUpdate })` watching `.hero__scroll-runway`, hand back one 0–1
 * `progress` number, smoothed by GSAP's own `scrub` lag rather than a
 * hand-rolled lerp. `renderForProgress` (below) turns that one number
 * into three eased sub-values — `crackT`, `hatchT`, `aftermathT`, one
 * per named phase (Tension / Hatch / Reveal) — and every visual system
 * (shell opacity, shard positions, the seam glow, the Vanguard Node,
 * the backdrop, the page's CSS theme) is just a different interpolation
 * of those three numbers. One function, one source of truth — nothing
 * can drift out of sync with anything else.
 *
 * INSTANCEDMESH: WHY, AND WHAT IT COSTS YOU.
 * `InstancedMesh` means ONE geometry, ONE material, ONE draw call, with
 * every instance's transform uploaded together as a single matrix buffer
 * (`setMatrixAt` + `instanceMatrix.needsUpdate`) — versus one draw call
 * PER shard if these were separate `THREE.Mesh` objects. The real cost:
 * every instance MUST share that one geometry — you can't give instance
 * #3 a different shape than instance #7.
 *
 * ELIMINATING THE "SWAP": WHY THE FACETS ARE BUILT FROM REAL ICOSAHEDRON
 * FACE DATA, NOT A SEPARATE SOLID MESH.
 * The previous version kept two different objects — a solid, flat-shaded
 * `IcosahedronGeometry` for the rest state, and a Fibonacci-sphere-
 * scattered set of unrelated pentagon "chips" for the fragments — cross-
 * fading between them once cracking began. That handoff between two
 * geometrically-unrelated shapes is exactly what read as "popping."
 * This version has only ONE object, ever: `buildFacetData` reads the
 * actual triangle data out of a real `THREE.IcosahedronGeometry(radius,
 * detail)` — for each of its faces, the 3 corner vertices, their
 * centroid, and a proper face normal (via the edge cross product) — and
 * uses that to place one `InstancedMesh` instance PER FACE, oriented so
 * its outward side exactly matches that face's real position and
 * normal. `shardGeometry` itself is a single equilateral triangle sized
 * to match a real face's edge length (Three.js's icosahedron subdivision
 * produces faces that are already all extremely close in size, a known
 * property of geodesic subdivision, so one representative triangle tiles
 * every face closely). At rest — "spacing" (`CRACK_RADIAL_SHIFT` /
 * `HATCH_RADIAL_SPREAD`, both zero at progress 0) — every instance sits
 * exactly on its real face, so the assembled InstancedMesh IS a solid
 * icosahedron, down to the vertex, with nothing to swap away from.
 *
 * THE CAMERA NEAR PLANE: WHY SHARDS VANISH, NOT FADE.
 * `camera.near` (0.1 here) is the literal distance, in front of the
 * camera along its view direction, closer than which the perspective
 * projection matrix renders NOTHING — it's one of the six planes
 * (near/far/left/right/top/bottom) that make up a camera's view
 * frustum, the same mechanism that already keeps you from seeing
 * objects behind the camera or miles past `camera.far`. The camera
 * sits at a fixed `CAMERA_REST_Z`; a shard's distance from it is
 * `CAMERA_REST_Z - shard.position.z`. As the Hatch phase drives that
 * shard's Z up past `CAMERA_REST_Z` (see `HATCH_Z_RUSH`, deliberately
 * larger than `CAMERA_REST_Z` itself), that distance shrinks toward
 * zero and keeps going negative. The exact instant it drops below
 * `camera.near`, every triangle of that shard is nearer than the near
 * plane — outside the frustum entirely — so the GPU rasterizer simply
 * does not draw it that frame. Nothing in this file explicitly hides
 * the shard for this; it's a hard, built-in consequence of how any
 * perspective camera already clips geometry, which is exactly why it
 * reads as a crisp "swoosh past you and gone" rather than a soft fade —
 * there is no fade at a clip plane by default. Combined with each
 * shard's `growScale` (deliberately enlarging it well beyond what
 * natural perspective alone would do as it approaches the lens — see
 * "grow larger as they approach" in the brief), a shard reads as
 * rushing up to fill part of the frame and then instantly vanishing,
 * rather than shrinking away into the distance. Once every shard is
 * comfortably past that point, the whole `InstancedMesh` is also set
 * `visible = false` — that's a separate, purely-for-performance cleanup
 * step (stop asking the GPU to test triangles that can't possibly be
 * on screen), not what makes them disappear in the first place.
 *
 * SELECTIVE BLOOM: WHY A LUMINANCE THRESHOLD ALONE WASN'T ENOUGH.
 * The first version of this bloom pass ran `UnrealBloomPass` over the
 * single, normal render — relying purely on its luminance threshold to
 * decide what glows. That leaked: the backdrop's own bright accent blobs
 * (there for the glass to refract, see paintDarkBackdrop) were bright
 * enough to bloom too, even at rest before anything had cracked — the
 * opposite of "the outer glass should stay dark ... until it breaks."
 * A threshold can't distinguish "bright because it's the Node" from
 * "bright because it's a light source behind the glass." Real selective
 * bloom needs two full render passes:
 *   1. `bloomComposer` renders the ENTIRE scene, but first every object
 *      NOT marked with `BLOOM_LAYER` (`darkenNonBloomed`, below) has its
 *      real material temporarily swapped for solid black, and
 *      `scene.background` is swapped for solid black too. Only the
 *      Vanguard Node and the crack-seam strips keep their real
 *      (bright) material. Rendering the full scene rather than only the
 *      glowing objects matters: a solid-black shell or shard in FRONT
 *      of the Node still correctly blocks/occludes it in this pass,
 *      which pure layer-filtering (hiding non-bloom objects entirely)
 *      would get wrong.
 *   2. That blackened render feeds `UnrealBloomPass`, producing a glow
 *      texture that is, by construction, black everywhere except the
 *      Node and the cracks.
 *   3. Materials and background are restored, the scene is rendered
 *      NORMALLY (full transmission/reflections/lighting intact) into
 *      `finalComposer`, and a `ShaderPass` additively blends that
 *      bloom texture on top. The glass shell and shards are never part
 *      of the bloom source, so they can't glow no matter how bright
 *      their specular highlights get — only what's actually on
 *      `BLOOM_LAYER` can.
 *
 * THE ENVIRONMENT MAP: PROCEDURAL, NOT A LOADED .hdr FILE.
 * `scene.environment` (below) is generated by rendering a small lit
 * scene through `PMREMGenerator`, not by fetching a real equirectangular
 * .hdr image via `RGBELoader`. Both are legitimate "environment maps" as
 * far as Three.js's lighting model is concerned — PMREMGenerator's job
 * IS to turn any scene (procedural or a loaded photo) into the
 * pre-filtered mip chain a PBR material samples for reflections; a
 * loaded HDR is just one more scene it could have processed. Given this
 * project's standing constraint (CDN-only, no bundler, no binary asset
 * pipeline), fetching a real HDR file would mean introducing an
 * `RGBELoader` import plus sourcing and hosting a multi-megabyte binary
 * asset for one accent object — a much larger dependency footprint than
 * "a basic EnvironmentMap" calls for. The environment below was widened
 * (three lights at different angles/colors instead of two) specifically
 * so the diamond material has enough directional variety to catch real
 * highlights rather than one flat reflection.
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

  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  var scene = new THREE.Scene();
  var CAMERA_REST_Z = 6;
  var camera = new THREE.PerspectiveCamera(35, 1, 0.1, 100);
  camera.position.set(0, 0, CAMERA_REST_Z);

  var scrollRunway = document.querySelector(".hero__scroll-runway");

  // --- Selective bloom: two full EffectComposers, not one — see the
  // Teacher Mode note above for why a plain threshold on a single render
  // wasn't actually selective. `BLOOM_LAYER` marks which objects are
  // allowed to glow (only the Vanguard Node + crack seams enable it,
  // further down); everything else gets darkened to solid black for
  // `bloomComposer`'s pass only, via `darkenNonBloomed`/`restoreMaterial`. ---
  var BLOOM_LAYER = 1;
  var bloomLayer = new THREE.Layers();
  bloomLayer.set(BLOOM_LAYER);

  var darkMaterial = new THREE.MeshBasicMaterial({ color: 0x000000 });
  var materialCache = {};
  function darkenNonBloomed(obj) {
    if (obj.isMesh && bloomLayer.test(obj.layers) === false) {
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

  var bloomPass = new UnrealBloomPass(new THREE.Vector2(1, 1), 1.2, 0.55, 0.1);
  var bloomComposer = new EffectComposer(renderer);
  bloomComposer.renderToScreen = false;
  bloomComposer.addPass(new RenderPass(scene, camera));
  bloomComposer.addPass(bloomPass);

  // Additively blends bloomComposer's (mostly-black, glow-only) output
  // on top of the normal, fully-lit render — see Teacher Mode step 3.
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
  var finalComposer = new EffectComposer(renderer);
  finalComposer.addPass(new RenderPass(scene, camera));
  finalComposer.addPass(mixPass);
  finalComposer.addPass(new OutputPass());

  var blackBackground = new THREE.Color(0x000000);

  function renderSelectiveBloom() {
    var realBackground = scene.background;
    scene.background = blackBackground;
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
  }

  // World-space half-width/half-height of the camera's view frustum at a
  // given distance from it — used to size the Vanguard Node so it can
  // genuinely "swallow the camera" (fill the screen) regardless of the
  // viewer's aspect ratio, instead of a hard-coded world-unit size that
  // would over- or under-shoot on different screens.
  function frustumSizeAtDistance(distance) {
    var heightHalf = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * distance;
    return { halfWidth: heightHalf * camera.aspect, halfHeight: heightHalf };
  }

  function seededRandom(seed) {
    var x = Math.sin(seed * 12.9898) * 43758.5453;
    return x - Math.floor(x);
  }

  function smoothstep(t) {
    t = Math.min(1, Math.max(0, t));
    return t * t * (3 - 2 * t);
  }

  // --- A small, self-contained "environment" for the glass to reflect —
  // see the Teacher Mode note above on why this is procedural rather
  // than a loaded .hdr file. Three lights at different angles/colors
  // (not two) so a diamond-cut material has enough directional variety
  // to catch distinct highlights rather than one flat reflection. Drives
  // only reflections (scene.environment, IBL specular highlights) — kept
  // deliberately separate from scene.background below, which is what
  // transmission actually sees through the glass. ---
  function buildEnvironmentMap() {
    var envScene = new THREE.Scene();

    var wallMaterial = new THREE.MeshBasicMaterial({ color: 0x20232e, side: THREE.BackSide });
    envScene.add(new THREE.Mesh(new THREE.SphereGeometry(6, 16, 16), wallMaterial));

    var warmLight = new THREE.PointLight(0xfff2e0, 11, 12);
    warmLight.position.set(3, 2, 3);
    envScene.add(warmLight);

    var coolLight = new THREE.PointLight(0x9fb2ff, 6, 12);
    coolLight.position.set(-3, -2, -3);
    envScene.add(coolLight);

    var rimLight = new THREE.PointLight(0xffffff, 7, 12);
    rimLight.position.set(-2, 4, -1);
    envScene.add(rimLight);

    var pmrem = new THREE.PMREMGenerator(renderer);
    var renderTarget = pmrem.fromScene(envScene, 0.04);
    pmrem.dispose();
    return renderTarget.texture;
  }

  scene.environment = buildEnvironmentMap();

  // --- The backdrop: a persistent offscreen 2D canvas, painted once at
  // startup (dark) and repainted (crossfaded toward light) only while
  // the Supernova is actually in progress — see paintBackdrop below.
  // Handed to Three.js as scene.background so transmission has real
  // content to bend through the glass shell. ---
  var BACKDROP_SIZE = 1024;
  var backdropCanvas = document.createElement("canvas");
  backdropCanvas.width = BACKDROP_SIZE;
  backdropCanvas.height = BACKDROP_SIZE;
  var backdropCtx = backdropCanvas.getContext("2d");
  var backdropTexture = new THREE.CanvasTexture(backdropCanvas);
  backdropTexture.colorSpace = THREE.SRGBColorSpace;
  scene.background = backdropTexture;

  function paintBlob(ctx, nx, ny, radiusFraction, color, alpha) {
    var cx = BACKDROP_SIZE * nx;
    var cy = BACKDROP_SIZE * ny;
    var r = BACKDROP_SIZE * radiusFraction;
    var gradient = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
    gradient.addColorStop(0, color);
    gradient.addColorStop(1, "rgba(0,0,0,0)");
    ctx.globalAlpha = alpha;
    ctx.fillStyle = gradient;
    ctx.fillRect(cx - r, cy - r, r * 2, r * 2);
  }

  function paintDarkBackdrop(ctx) {
    ctx.globalAlpha = 1;
    ctx.fillStyle = "#0d0d0f"; // matches --color-bg
    ctx.fillRect(0, 0, BACKDROP_SIZE, BACKDROP_SIZE);

    ctx.globalCompositeOperation = "lighter";
    paintBlob(ctx, 0.2, 0.3, 0.42, "#171922", 1);
    paintBlob(ctx, 0.8, 0.2, 0.4, "#10121e", 1);
    paintBlob(ctx, 0.7, 0.85, 0.45, "#1a1c24", 1);
    paintBlob(ctx, 0.25, 0.8, 0.4, "#101115", 1);
    // Bright glows — what gives the glass something to refract, echoing
    // the object's own light sources (keyLight, the env map's coolLight,
    // the Vanguard Node's blue) so the backdrop reads as "those same
    // lights, seen bent through the glass."
    paintBlob(ctx, 0.78, 0.22, 0.3, "#fff4e8", 0.55);
    paintBlob(ctx, 0.15, 0.75, 0.28, "#9fb2ff", 0.4);
    paintBlob(ctx, 0.5, 0.5, 0.22, "#3d8bff", 0.3);
    ctx.globalCompositeOperation = "source-over";
  }

  function paintLightBackdrop(ctx, t) {
    if (t <= 0) return;
    ctx.globalAlpha = t;
    ctx.fillStyle = "#eef4fb"; // bright, "effortless" Vanguard blue/white
    ctx.fillRect(0, 0, BACKDROP_SIZE, BACKDROP_SIZE);

    ctx.globalCompositeOperation = "lighter";
    paintBlob(ctx, 0.25, 0.3, 0.5, "#ffffff", t * 0.8);
    paintBlob(ctx, 0.75, 0.7, 0.55, "#dceafd", t * 0.7);
    ctx.globalCompositeOperation = "source-over";
  }

  var lastPaintedThemeT = -1;
  function paintBackdrop(aftermathT) {
    if (Math.abs(aftermathT - lastPaintedThemeT) < 0.002) return;
    paintDarkBackdrop(backdropCtx);
    paintLightBackdrop(backdropCtx, aftermathT);
    backdropTexture.needsUpdate = true;
    lastPaintedThemeT = aftermathT;
  }
  paintBackdrop(0);

  var ambient = new THREE.AmbientLight(0xffffff, 0.25);
  scene.add(ambient);

  var keyLight = new THREE.PointLight(0xfff4e8, 6, 20);
  keyLight.position.set(2, 2, 4);
  scene.add(keyLight);

  var VANGUARD_BLUE = new THREE.Color(0x3d8bff);
  var SUPERNOVA_WHITE = new THREE.Color(0xffffff);

  // --- Shared "diamond" material for every facet instance. ior: 2.417
  // is the actual refractive index of diamond (glass is ~1.5) — a much
  // more aggressive bend of the backdrop behind it, which combined with
  // thickness: 2.5 (more physical depth to refract through) and
  // roughness: 0.05 (near-mirror smooth) is what separates "glass ball"
  // from "cut diamond." transmission: 1 keeps it fully clear rather than
  // frosted. ---
  var material = new THREE.MeshPhysicalMaterial({
    color: 0xffffff,
    metalness: 0,
    roughness: 0.05,
    transmission: 1.0,
    thickness: 2.5,
    ior: 2.417,
    envMapIntensity: 1.4,
    clearcoat: 1,
    clearcoatRoughness: 0.05,
    transparent: true,
    opacity: 1,
  });

  var shellGroup = new THREE.Group(); // fixed at the origin — no more idle-corner/Snap this time, see Teacher Mode
  scene.add(shellGroup);

  var tiltGroup = new THREE.Group(); // owned by mouse-tilt
  shellGroup.add(tiltGroup);

  var SHELL_RADIUS = 1.0;
  var SHARD_DEPTH = 0.14;

  // --- Read real face data out of an actual IcosahedronGeometry — see
  // the Teacher Mode note above ("ELIMINATING THE SWAP") for why this
  // replaces the previous separate-solid-mesh + Fibonacci-sphere-chip
  // approach entirely. detail: 1 → 80 faces: large enough to read as
  // "sharp, faceted diamond faces," not so many that individual facets
  // disappear into a smooth-looking blur. The geometry is only used here,
  // at startup, to extract data — it's disposed immediately after. ---
  function buildFacetData(radius, detail) {
    var geo = new THREE.IcosahedronGeometry(radius, detail);
    var pos = geo.attributes.position;
    var faces = [];
    var a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
    for (var idx = 0; idx < pos.count; idx += 3) {
      a.fromBufferAttribute(pos, idx);
      b.fromBufferAttribute(pos, idx + 1);
      c.fromBufferAttribute(pos, idx + 2);

      var centroid = new THREE.Vector3().add(a).add(b).add(c).divideScalar(3);
      var normal = new THREE.Vector3()
        .crossVectors(new THREE.Vector3().subVectors(b, a), new THREE.Vector3().subVectors(c, a))
        .normalize();
      if (normal.dot(centroid) < 0) normal.negate(); // ensure outward-facing

      var tangent = new THREE.Vector3().subVectors(a, centroid).normalize();

      faces.push({ centroid: centroid, normal: normal, tangent: tangent, edgeLength: a.distanceTo(b) });
    }
    geo.dispose();
    return faces;
  }

  var FACET_DETAIL = 1;
  var facets = buildFacetData(SHELL_RADIUS, FACET_DETAIL);
  var SHARD_COUNT = facets.length;

  var averageEdgeLength =
    facets.reduce(function (sum, f) { return sum + f.edgeLength; }, 0) / facets.length;
  var averageFacetRadius =
    facets.reduce(function (sum, f) { return sum + f.centroid.length(); }, 0) / facets.length;

  // One representative equilateral triangle, sized to a real face's edge
  // length — every instance shares this exact geometry (see Teacher Mode
  // note on InstancedMesh's one-geometry constraint). Because icosahedron
  // subdivision produces near-congruent faces, one shape tiles every real
  // face closely — but the IN-PLANE rotation matters and must match the
  // basis below exactly: vertex 0 sits at local angle 0 (local +X)
  // because `shardData`'s basis matrix maps local +X to `projectedTangent`
  // (the real direction from each face's centroid to its own vertex A).
  // Any other starting angle that isn't a multiple of 120° — this
  // shipped with vertex 0 at local +Y (90°) first, which is NOT a
  // multiple of 120°, and produced a visibly wrong, chaotic-looking
  // tiling instead of a solid diamond — silently misaligns every facet.
  function createFacetGeometry(edgeLength, depth) {
    var shape = new THREE.Shape();
    var circumradius = edgeLength / Math.sqrt(3);
    for (var k = 0; k < 3; k++) {
      var angle = k * ((Math.PI * 2) / 3);
      var x = Math.cos(angle) * circumradius;
      var y = Math.sin(angle) * circumradius;
      if (k === 0) shape.moveTo(x, y);
      else shape.lineTo(x, y);
    }
    shape.closePath();
    var bevel = Math.min(depth * 0.3, 0.015);
    var geo = new THREE.ExtrudeGeometry(shape, {
      depth: depth,
      bevelEnabled: true,
      bevelThickness: bevel,
      bevelSize: bevel,
      bevelSegments: 2,
      curveSegments: 1,
    });
    geo.translate(0, 0, -depth / 2);
    return geo;
  }

  var shardGeometry = createFacetGeometry(averageEdgeLength, SHARD_DEPTH);
  var shardMesh = new THREE.InstancedMesh(shardGeometry, material, SHARD_COUNT);
  shardMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  tiltGroup.add(shardMesh);

  // Breathing: a real GSAP loop tween (yoyo + infinite repeat) on the
  // InstancedMesh's own scale — "slow scale loop" is literally what this
  // GSAP feature is for. Skipped under reduced motion like every other
  // purely-ambient animation in this file. There's no separate solid
  // mesh to breathe anymore; the InstancedMesh IS the shell, from frame 1.
  if (!prefersReducedMotion) {
    gsap.to(shardMesh.scale, {
      x: 1.045,
      y: 1.045,
      z: 1.045,
      duration: 3.2,
      ease: "sine.inOut",
      yoyo: true,
      repeat: -1,
    });
  }

  var basisMatrix = new THREE.Matrix4();
  var bitangent = new THREE.Vector3();
  var projectedTangent = new THREE.Vector3();
  var shardData = facets.map(function (face, i) {
    projectedTangent.copy(face.tangent).addScaledVector(face.normal, -face.tangent.dot(face.normal)).normalize();
    bitangent.crossVectors(face.normal, projectedTangent).normalize();
    basisMatrix.makeBasis(projectedTangent, bitangent, face.normal);
    var restQuaternion = new THREE.Quaternion().setFromRotationMatrix(basisMatrix);

    return {
      dir: face.normal.clone(),
      restPosition: face.centroid.clone(),
      restQuaternion: restQuaternion,
      speedFactor: 0.7 + seededRandom(i + 600) * 0.6,
      tumbleAxis: new THREE.Vector3(
        seededRandom(i + 800) - 0.5,
        seededRandom(i + 850) - 0.5,
        seededRandom(i + 900) - 0.5
      ).normalize(),
      tumbleSpeed: 0.5 + seededRandom(i + 950) * 1.5,
    };
  });

  // --- Light seepage: three ring "seams" (a beach-ball-style tri-band
  // pattern) sit just outside the facet surface at rest, barely visible
  // — then during the Tension phase, these same rings are what "light
  // seeps and bleeds" through as the facets begin to drift apart,
  // brightening via crackGlow in renderForProgress and lit up by
  // UnrealBloomPass once they cross onto BLOOM_LAYER. Radius is derived
  // from the facets' own actual (measured, not assumed) average radius —
  // flat facets sit slightly inside the nominal sphere radius (a chord is
  // always shorter than the arc it spans), so hugging the real geometry
  // instead of SHELL_RADIUS keeps these seams from floating visibly
  // outside the diamond's real silhouette. ---
  var SEAM_RADIUS = averageFacetRadius * 1.015;
  var seamRotations = [
    new THREE.Euler(0, 0, 0),
    new THREE.Euler(Math.PI / 3, Math.PI / 4, 0),
    new THREE.Euler(-Math.PI / 3, -Math.PI / 4, Math.PI / 6),
  ];
  var crackSeams = seamRotations.map(function (rot) {
    var geo = new THREE.TorusGeometry(SEAM_RADIUS, 0.012, 8, 64);
    var mat = new THREE.MeshBasicMaterial({
      color: VANGUARD_BLUE,
      transparent: true,
      opacity: 0.12,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    var mesh = new THREE.Mesh(geo, mat);
    mesh.rotation.copy(rot);
    mesh.layers.enable(BLOOM_LAYER); // light seeping through the crack IS allowed to glow
    tiltGroup.add(mesh);
    return mesh;
  });

  // --- The Vanguard Node: the glowing wireframe sphere the Hatch
  // reveals. A crisp wireframe icosahedron plus a larger additive-
  // blended shell standing in for a glow/bloom source (now genuinely
  // amplified by UnrealBloomPass, not just faked with layered
  // transparency), plus a small point light so it casts a faint colored
  // bounce onto the shards around it. Crossfades from Vanguard Blue to
  // white as the Supernova "swallows the camera." ---
  var nodeColorScratch = new THREE.Color();

  var nodeGroup = new THREE.Group();

  // depthTest: true (explicit, not just relying on the default) — the
  // Node must stay properly occluded by the facets in front of it rather
  // than drawing through them; combined with NODE_CONTAINMENT below,
  // this is what keeps it visually sealed inside the glass until the
  // shell actually opens up.
  var nodeWireMaterial = new THREE.MeshBasicMaterial({
    color: VANGUARD_BLUE,
    wireframe: true,
    transparent: true,
    opacity: 0.5,
    depthTest: true,
  });
  var nodeWire = new THREE.Mesh(new THREE.IcosahedronGeometry(0.22, 1), nodeWireMaterial);
  nodeWire.layers.enable(BLOOM_LAYER);
  nodeGroup.add(nodeWire);

  var nodeGlowMaterial = new THREE.MeshBasicMaterial({
    color: VANGUARD_BLUE,
    transparent: true,
    opacity: 0.15,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    depthTest: true,
  });
  var nodeGlow = new THREE.Mesh(new THREE.IcosahedronGeometry(0.34, 1), nodeGlowMaterial);
  nodeGlow.layers.enable(BLOOM_LAYER);
  nodeGroup.add(nodeGlow);

  var nodeLight = new THREE.PointLight(VANGUARD_BLUE.getHex(), 0.5, 3);
  nodeGroup.add(nodeLight);

  tiltGroup.add(nodeGroup);

  // --- Timeline boundaries, matching the brief's exact percentages:
  // Tension 0–30%, Hatch 30–60%, Reveal/Supernova 60–100%. No offset
  // gap this time — there's no Snap tween to let settle first, so the
  // crack can begin at progress 0 exactly as described. ---
  var CRACK_END = 0.3;
  var HATCH_END = 0.6;

  var CRACK_RADIAL_SHIFT = 0.12;
  var HATCH_RADIAL_SPREAD = 2.6;
  var HATCH_Z_RUSH = 11; // > CAMERA_REST_Z — see Teacher Mode note on the near plane
  var GROW_FACTOR = 1.8; // shards grow up to +180% size approaching the lens

  var CORE_BASE_REVEAL = 0.04; // near-invisible at rest — "a singular, heavy piece of glass"
  // Hard containment cap on the Node's size before the Supernova: keeps
  // its outer glow radius at or under 0.85x the shell's own radius while
  // it's still meant to be sealed inside the glass, so it can never
  // visibly poke past the facets — only the Aftermath's growth to
  // fillScale (below) is allowed to exceed the shell.
  var NODE_CONTAINMENT_SCALE = 0.85;

  var DARK_BG = "#0d0d0f";
  var LIGHT_BG = "#eef4fb";
  var DARK_TEXT_PRIMARY = "#f5f5f2";
  var LIGHT_TEXT_PRIMARY = "#23262b"; // dark charcoal
  var DARK_TEXT_MUTED = "#9a9aa1";
  var LIGHT_TEXT_MUTED = "#565c66";

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

  var root = document.documentElement;
  var scratchMatrix = new THREE.Matrix4();
  var scratchQuat = new THREE.Quaternion();
  var scratchScale = new THREE.Vector3();
  var scratchPos = new THREE.Vector3();

  var lastCrackT = 0;
  var lastHatchT = 0;

  // The single per-frame fan-out described in the Teacher Mode note
  // above: one scroll-timeline progress number in, every visual system
  // updated out. Pure function of `progress` — no reliance on elapsed
  // time — so scrubbing forward or backward always lands on the same
  // state, which a real velocity/physics simulation could not guarantee.
  function renderForProgress(progress) {
    var crackT = smoothstep(progress / CRACK_END);
    var hatchT = smoothstep((progress - CRACK_END) / (HATCH_END - CRACK_END));
    var aftermathT = smoothstep((progress - HATCH_END) / (1 - HATCH_END));
    lastCrackT = crackT;
    lastHatchT = hatchT;

    // Cubic ease-in — see "AN EASED POSITION FUNCTION, NOT A PHYSICS
    // SIM" from the earlier Monolith pass's Teacher Mode note, still the
    // right tool here: tiny position deltas early in the Hatch, much
    // larger ones late, which is what "burst of speed" actually means.
    var hatchBurst = hatchT * hatchT * hatchT;

    // No separate solid shell to fade — the InstancedMesh IS the shell,
    // from frame 1 (see Teacher Mode note on eliminating the swap). It
    // fades out only at the very end, once the Supernova has fully
    // bloomed, and is then actually removed from rendering rather than
    // left invisible-via-alpha.
    material.opacity = 1 - aftermathT;
    shardMesh.visible = material.opacity > 0.015;

    if (shardMesh.visible) {
      for (var s = 0; s < SHARD_COUNT; s++) {
        var d = shardData[s];
        // "Spacing": zero at progress 0 (crackT and hatchBurst both 0),
        // so every facet sits exactly on its real rest position/rotation
        // — a perfectly assembled, gapless diamond, not an approximation.
        var radialSpread = crackT * CRACK_RADIAL_SHIFT + hatchBurst * HATCH_RADIAL_SPREAD * d.speedFactor;
        var zRush = hatchBurst * HATCH_Z_RUSH * d.speedFactor;

        scratchPos.set(
          d.restPosition.x + d.dir.x * radialSpread,
          d.restPosition.y + d.dir.y * radialSpread,
          d.restPosition.z + d.dir.z * radialSpread + zRush
        );

        var tumbleAngle = (crackT * 0.3 + hatchT * 2.0) * d.tumbleSpeed;
        scratchQuat.copy(d.restQuaternion).multiply(
          new THREE.Quaternion().setFromAxisAngle(d.tumbleAxis, tumbleAngle)
        );

        var growScale = 1 + hatchBurst * GROW_FACTOR * d.speedFactor;
        scratchScale.setScalar(growScale);

        scratchMatrix.compose(scratchPos, scratchQuat, scratchScale);
        shardMesh.setMatrixAt(s, scratchMatrix);
      }
      shardMesh.instanceMatrix.needsUpdate = true;
    }

    // The Fracture's light seepage: a faint baseline (the "beveled seam
    // lines" visible even at rest) plus a bright bloom that peaks mid-
    // Tension and fades once the Hatch takes over, then fully vanishes
    // through the Supernova so no stray seam geometry lingers once the
    // screen has gone white.
    var crackGlow = crackT * (1 - hatchT);
    var seamOpacity = (0.12 + crackGlow * 0.8) * (1 - aftermathT);
    var seamsVisible = seamOpacity > 0.01;
    nodeColorScratch.copy(VANGUARD_BLUE).lerp(SUPERNOVA_WHITE, Math.min(1, crackGlow * 1.4));
    crackSeams.forEach(function (seam) {
      seam.visible = seamsVisible;
      if (!seamsVisible) return;
      seam.material.opacity = seamOpacity;
      seam.material.color.copy(nodeColorScratch);
    });

    // Vanguard Node: a bright hint appears as the crack opens, brightens
    // through the Hatch, then blooms to fill the screen (the Supernova
    // "swallowing the camera") through the Reveal. Fill-scale is derived
    // from the camera's own frustum, not a hard-coded number, so it
    // actually fills regardless of viewport shape.
    var reveal = CORE_BASE_REVEAL + (1 - CORE_BASE_REVEAL) * Math.min(1, crackT * 0.5 + hatchT * 0.5);
    var containedScale = (1 + crackT * 0.15 + hatchT * 0.7) * NODE_CONTAINMENT_SCALE;
    var frustum = frustumSizeAtDistance(CAMERA_REST_Z);
    var fillScale = (Math.max(frustum.halfWidth, frustum.halfHeight) * 1.15) / 0.34; // 0.34 == nodeGlow's base radius
    var nodeScale = containedScale + (fillScale - containedScale) * aftermathT;
    nodeGroup.scale.setScalar(nodeScale);

    nodeWireMaterial.opacity = 0.25 + 0.65 * reveal;
    nodeGlowMaterial.opacity = Math.min(1, 0.06 + 0.4 * reveal + aftermathT * 0.6);
    nodeLight.intensity = reveal * 3 + aftermathT * 4;

    var nodeColor = VANGUARD_BLUE.clone().lerp(SUPERNOVA_WHITE, aftermathT);
    nodeWireMaterial.color.copy(nodeColor);
    nodeGlowMaterial.color.copy(nodeColor);
    nodeLight.color.copy(nodeColor);

    // Page theme: the WebGL backdrop and the CSS custom properties are
    // both driven by this same aftermathT, so the 3D scene and the DOM
    // around it cross the dark → light threshold at exactly the same
    // scroll position.
    paintBackdrop(aftermathT);
    root.style.setProperty("--color-bg", lerpColorString(DARK_BG, LIGHT_BG, aftermathT));
    root.style.setProperty("--color-text-primary", lerpColorString(DARK_TEXT_PRIMARY, LIGHT_TEXT_PRIMARY, aftermathT));
    root.style.setProperty("--color-text-muted", lerpColorString(DARK_TEXT_MUTED, LIGHT_TEXT_MUTED, aftermathT));
  }

  // --- Cursor tracking for the subtle tilt + key-light nudge. Same
  // exponential-smoothing technique as js/main.js's grain/glow drift. ---
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

  // --- The render loop, driven by GSAP's own ticker rather than a
  // separate requestAnimationFrame call — GSAP's tweens (the breathing
  // loop) and ScrollTrigger's scrub both already run on gsap's internal
  // ticker, so rendering from that same ticker keeps everything on one
  // shared clock instead of two independent rAF loops racing each other. ---
  gsap.ticker.add(function () {
    var now = performance.now();
    renderForProgress(mainProgress);

    if (!prefersReducedMotion) {
      tiltCurrentX += (tiltTargetX - tiltCurrentX) * TILT_SMOOTHING;
      tiltCurrentY += (tiltTargetY - tiltCurrentY) * TILT_SMOOTHING;
      tiltGroup.rotation.set(tiltCurrentX * 0.15, tiltCurrentY * 0.15, 0);

      keyLight.position.x = 2 + tiltCurrentY * 1.2;
      keyLight.position.y = 2 - tiltCurrentX * 1.2;

      // Node pulse speeds up with tension — calm at rest, agitated once
      // the crack opens and through the Hatch's burst.
      var pulseSpeed = 0.0022 + lastCrackT * 0.0035 + lastHatchT * 0.006;
      var nodePulse = 1 + Math.sin(now * pulseSpeed) * 0.1;
      nodeWire.scale.setScalar(nodePulse);
      nodeGlow.scale.setScalar(nodePulse);
    }

    renderSelectiveBloom();
  });
})();
