// Deterministic (seeded) initial-position generator for the dust cloud.
// Shared by gpgpu/useDustSimulation.js (which bakes these into the
// GPGPU position texture) and components/canvas/DataLines.jsx (which
// needs a CPU-readable copy of the SAME positions to compute
// nearest-neighbor pairs at mount time — see that file's Teacher Mode
// note). Using `Math.random()` independently in both places would give
// each one a different cloud, so the "connections between nearby
// particles" would point at the wrong particles; a seeded PRNG makes
// both call sites reproduce the identical distribution.

function mulberry32(seed) {
  let state = seed >>> 0;
  return function random() {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const DUST_SEED = 1337;

/**
 * @returns {Float32Array} flat RGBA-per-particle layout — [x, y, z, seed, ...]
 * — matching the GPGPU position texture's texel layout 1:1, including the
 * padding texels beyond `count` (parked far away, see useDustSimulation.js).
 */
export function generateDustPositions(count, textureSize, seed = DUST_SEED) {
  const random = mulberry32(seed);
  const total = textureSize * textureSize;
  const data = new Float32Array(total * 4);

  for (let i = 0; i < total; i++) {
    const s = i * 4;
    if (i < count) {
      // Soft-shell diffuse volume: bias radius sampling with sqrt so
      // density falls off gently instead of a flat/uniform-looking ball.
      const radius = 3.5 + Math.pow(random(), 0.5) * 14;
      const theta = random() * Math.PI * 2;
      const phi = Math.acos(2 * random() - 1);
      data[s] = radius * Math.sin(phi) * Math.cos(theta);
      data[s + 1] = radius * Math.sin(phi) * Math.sin(theta);
      data[s + 2] = radius * Math.cos(phi);
      data[s + 3] = random(); // per-particle seed (size/color variance in the render shader)
    } else {
      data[s] = 9999;
      data[s + 1] = 9999;
      data[s + 2] = 9999;
      data[s + 3] = 0;
    }
  }

  return data;
}
