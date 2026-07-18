import { useMemo, useRef } from "react";
import * as THREE from "three";
import { useFrame } from "@react-three/fiber";
import { generateDustPositions } from "../../lib/particleDistribution";
import { DATA_LINE_MAX_SEGMENTS, PALETTE } from "../../lib/constants";
import { timelineUniforms } from "../../hooks/useScrollTimeline";
import vertexShader from "../../shaders/dataLines/dataLines.vert.glsl?raw";
import fragmentShader from "../../shaders/dataLines/dataLines.frag.glsl?raw";

/**
 * Teacher Mode — faint data-line connections between nearby particles,
 * WITHOUT a per-frame nearest-neighbor search. A true kNN pass across
 * 10-20k particles, every frame, has no cheap GPU primitive in plain
 * WebGL (it's an O(n²) comparison problem) — so instead, `buildNeighborPairs`
 * runs ONCE, on the CPU, at mount, using a coarse spatial hash grid to
 * find each sampled particle's nearest neighbor in roughly O(n) rather
 * than O(n²). That gives a fixed TOPOLOGY: a capped list of (i, j)
 * particle-index pairs. What's NOT fixed is where those particles
 * currently are — each line vertex carries only its endpoint's index,
 * and the vertex shader looks up that particle's CURRENT position from
 * the exact same GPGPU position texture the dust cloud itself reads
 * (`gpu`, owned and stepped once per frame by the parent DustSystem.jsx)
 * — so every line's endpoints move with the live simulation for free,
 * with zero per-frame CPU cost, even though the pairing decision itself
 * was made once and never revisited.
 *
 * Uses the SAME seeded generator (lib/particleDistribution.js) that
 * seeds the GPGPU texture, so the pairs computed here from a CPU-side
 * array actually correspond to where the real particles started.
 */
export default function DataLines({ gpu, count, textureSize }) {
  const geometry = useMemo(() => {
    const positions = generateDustPositions(count, textureSize);
    const pairIndices = buildNeighborPairs(positions, count, DATA_LINE_MAX_SEGMENTS);

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("aParticleIndex", new THREE.BufferAttribute(new Float32Array(pairIndices), 1));
    // Three.js's renderer derives the non-indexed draw count from
    // `geometry.attributes.position.count` — required even though this
    // shader never reads `position`'s values (see dataLines.vert.glsl).
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(pairIndices.length * 3), 3));
    return geo;
  }, [count, textureSize]);

  const material = useMemo(
    () =>
      new THREE.RawShaderMaterial({
        glslVersion: THREE.GLSL3,
        vertexShader,
        fragmentShader,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        uniforms: {
          uPositionTexture: { value: null },
          uTextureSize: { value: textureSize },
          uColor: { value: new THREE.Color(PALETTE.teal) },
          uOpacity: { value: 0.16 },
          uFlash: { value: 0 },
        },
      }),
    [textureSize]
  );

  const linesRef = useRef();

  useFrame(() => {
    material.uniforms.uPositionTexture.value = gpu.getPositionTexture();
    material.uniforms.uFlash.value = timelineUniforms.flash;
  });

  return <lineSegments ref={linesRef} geometry={geometry} material={material} frustumCulled={false} />;
}

function buildNeighborPairs(positions, count, maxSegments) {
  const cellSize = 2.2;
  const grid = new Map();
  const cellKey = (x, y, z) => `${Math.floor(x / cellSize)}|${Math.floor(y / cellSize)}|${Math.floor(z / cellSize)}`;

  for (let i = 0; i < count; i++) {
    const s = i * 4;
    const key = cellKey(positions[s], positions[s + 1], positions[s + 2]);
    let bucket = grid.get(key);
    if (!bucket) {
      bucket = [];
      grid.set(key, bucket);
    }
    bucket.push(i);
  }

  const pairs = [];
  const seenPairs = new Set();
  const maxReach = (cellSize * 1.5) ** 2;
  // Stride through particles rather than visiting every one — caps total
  // work and total segment count together in one pass.
  const stride = Math.max(1, Math.floor(count / (maxSegments * 1.4)));

  for (let i = 0; i < count && pairs.length / 2 < maxSegments; i += stride) {
    const s = i * 4;
    const x = positions[s];
    const y = positions[s + 1];
    const z = positions[s + 2];
    const cx = Math.floor(x / cellSize);
    const cy = Math.floor(y / cellSize);
    const cz = Math.floor(z / cellSize);

    let bestJ = -1;
    let bestDist = Infinity;

    for (let ox = -1; ox <= 1; ox++) {
      for (let oy = -1; oy <= 1; oy++) {
        for (let oz = -1; oz <= 1; oz++) {
          const bucket = grid.get(`${cx + ox}|${cy + oy}|${cz + oz}`);
          if (!bucket) continue;
          for (const j of bucket) {
            if (j === i) continue;
            const pairKey = i < j ? `${i}_${j}` : `${j}_${i}`;
            if (seenPairs.has(pairKey)) continue;
            const js = j * 4;
            const dx = positions[js] - x;
            const dy = positions[js + 1] - y;
            const dz = positions[js + 2] - z;
            const dist = dx * dx + dy * dy + dz * dz;
            if (dist < bestDist) {
              bestDist = dist;
              bestJ = j;
            }
          }
        }
      }
    }

    if (bestJ !== -1 && bestDist < maxReach) {
      const pairKey = i < bestJ ? `${i}_${bestJ}` : `${bestJ}_${i}`;
      seenPairs.add(pairKey);
      pairs.push(i, bestJ);
    }
  }

  return pairs;
}
