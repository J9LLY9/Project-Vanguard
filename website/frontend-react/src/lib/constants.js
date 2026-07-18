// Single source of truth for the palette, act boundaries, and particle
// counts — every shader/component reads from here rather than
// hard-coding its own copy, so a tuning pass only ever touches one file.

// --- Palette (strict: obsidian / deep teal / white — no purples, no
// oranges, no "outer space" starfield colors; this is "digital/neural
// space"). Exported both as hex (for CSS/DOM) and as [r,g,b] 0..1
// (for shader uniforms, which take THREE.Color / vec3, not hex strings). ---
export const PALETTE = {
  obsidian: "#0a0a0a",
  obsidianDeep: "#050505",
  teal: "#0f6e56",
  tealBright: "#1fcf9e",
  white: "#ffffff",
  skyBlue: "#eaf6ff",
};

// --- Act boundaries, as fractions of the PINNED scroll region (Acts
// 1-3 only — see PinnedActs.jsx). Act 4/5 scroll natively and don't use
// these. Kept as plain fractions (not GSAP labels) so any component can
// cheaply remap the master `pinnedProgress` (0..1) into an act-local
// `t` via `(pinnedProgress - start) / (end - start)`, clamped. ---
export const ACT_BOUNDS = {
  singularity: { start: 0, end: 0.32 },
  eventHorizon: { start: 0.32, end: 0.72 },
  supernova: { start: 0.72, end: 1 },
};

// --- Particle tiers. `probeFpsTier` (see lib/fpsTier.js) picks one of
// these after a ~1s measurement window on load. Counts are chosen so
// each tier's simulation texture is a clean power-of-two square
// (GPUComputationRenderer wants width*height >= count, and a
// power-of-two texture is the cheapest to allocate/sample). ---
export const PARTICLE_TIERS = {
  high: { dust: 20000, textureSize: 142 }, // 142*142 = 20164
  mid: { dust: 12000, textureSize: 110 }, // 110*110 = 12100
  low: { dust: 6000, textureSize: 78 }, // 78*78 = 6084
  mobile: { dust: 2500, textureSize: 50 }, // 50*50 = 2500
};

export const DATA_LINE_MAX_SEGMENTS = 3200;
export const TAGLINE_PARTICLE_COUNT = 5000;
export const TAGLINE_TEXT = "MAKE LOCAL AI EFFORTLESS";

export const NAV_LINKS = [
  { label: "Product", href: "#product" },
  { label: "Benchmarks", href: "#benchmarks" },
  { label: "Docs", href: "#docs" },
  { label: "Get Started", href: "#get-started" },
];
