// Renders the tagline to an offscreen 2D canvas and samples it down to a
// point cloud — "the tagline text rendered as a particle field sampled
// from canvas-rendered typography, not DOM text" (Act 2 requirement).
// This runs ONCE on mount (see components/canvas/TaglineParticles.jsx),
// not per frame — it produces a static Float32Array of base positions
// that the shader then animates (pulled toward center, stretched by
// velocity) entirely on the GPU.

const CANVAS_WIDTH = 2048;
const CANVAS_HEIGHT = 512;

/**
 * @param {string} text
 * @param {number} targetCount how many particles to resample down to
 * @param {number} worldWidth world-space width the text should span
 * @returns {Float32Array} flat [x, y, z, x, y, z, ...] in world space, centered on origin
 */
export function sampleTextToParticles(text, targetCount, worldWidth = 10) {
  const canvas = document.createElement("canvas");
  canvas.width = CANVAS_WIDTH;
  canvas.height = CANVAS_HEIGHT;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });

  ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
  ctx.fillStyle = "#ffffff";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  // Shrink the font until the text fits within the canvas with a margin
  // — a fixed font size would clip or float tiny depending on the string.
  let fontSize = 320;
  ctx.font = `700 ${fontSize}px "Helvetica Neue", Arial, sans-serif`;
  const margin = CANVAS_WIDTH * 0.06;
  while (ctx.measureText(text).width > CANVAS_WIDTH - margin * 2 && fontSize > 20) {
    fontSize -= 4;
    ctx.font = `700 ${fontSize}px "Helvetica Neue", Arial, sans-serif`;
  }
  ctx.fillText(text, CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2);

  const { data } = ctx.getImageData(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

  // Collect every opaque pixel's coordinate first, THEN subsample — a
  // fixed stride alone would bias toward whichever glyph regions happen
  // to align with the stride, so we gather all candidates and take an
  // evenly-spaced slice through the shuffled set instead.
  const candidates = [];
  const stride = 2; // skip every other pixel while scanning — 4x fewer candidates, still dense enough
  for (let y = 0; y < CANVAS_HEIGHT; y += stride) {
    for (let x = 0; x < CANVAS_WIDTH; x += stride) {
      const alpha = data[(y * CANVAS_WIDTH + x) * 4 + 3];
      if (alpha > 128) candidates.push(x, y);
    }
  }

  const candidateCount = candidates.length / 2;
  const positions = new Float32Array(targetCount * 3);
  const worldHeight = worldWidth * (CANVAS_HEIGHT / CANVAS_WIDTH);

  for (let i = 0; i < targetCount; i++) {
    // Evenly-spaced index walk through the candidate list rather than
    // `Math.random()` picks — avoids clumping/holes from RNG variance
    // and is fully deterministic (same tagline always samples the same
    // way, which matters for visual QA).
    const srcIndex = candidateCount > 0 ? Math.floor((i / targetCount) * candidateCount) : 0;
    const px = candidates[srcIndex * 2] ?? CANVAS_WIDTH / 2;
    const py = candidates[srcIndex * 2 + 1] ?? CANVAS_HEIGHT / 2;

    const nx = px / CANVAS_WIDTH - 0.5; // -0.5..0.5
    const ny = 0.5 - py / CANVAS_HEIGHT; // flip Y: canvas is top-down, world is bottom-up

    positions[i * 3] = nx * worldWidth;
    positions[i * 3 + 1] = ny * worldHeight;
    positions[i * 3 + 2] = (Math.random() - 0.5) * 0.15; // faint depth jitter, not flat-flat
  }

  return positions;
}
