// Single source of truth for the palette and top-level nav — components
// read from here rather than hard-coding their own copy.

// Strictly monochrome: charcoal / grays / off-white — no hue anywhere,
// including the Three.js wireframe materials that read `accent`/
// `accentDim` below (those are plain JS values, not CSS, so they don't
// pick up anything from styles/global.css's custom properties — this
// file is the only source both sides actually share).
export const PALETTE = {
  obsidian: "#1d1d1f",
  accent: "#f5f5f7",
  accentDim: "#8e8e93",
  textPrimary: "#f5f5f7",
  textMuted: "#a1a1a6",
};

export const NAV_LINKS = [
  { label: "Product", href: "#product" },
  { label: "Benchmarks", href: "#benchmarks" },
  { label: "Docs", href: "#docs" },
  { label: "Get Started", href: "#get-started" },
];

export const GITHUB_URL = "https://github.com/J9LLY9/Project-Vanguard";
