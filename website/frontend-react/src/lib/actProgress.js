// Shared by the Act 1-3 DOM caption overlays: fades a caption in/out
// across the first/last 8% of its own act's range within the pinned
// scroll region, using the SAME `ACT_BOUNDS` (lib/constants.js) the
// GSAP timeline and the shaders derive gravity/flash from.
export function getActOpacity(pinnedProgress, bounds, fadeFraction = 0.08) {
  const { start, end } = bounds;
  const span = end - start;
  if (span <= 0) return 0;
  const t = (pinnedProgress - start) / span;
  if (t < 0 || t > 1) return 0;
  return Math.min(1, Math.min(t / fadeFraction, (1 - t) / fadeFraction));
}
