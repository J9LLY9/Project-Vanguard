// Act 2 intentionally has no DOM caption: the brief is explicit that the
// tagline is "rendered as a particle field sampled from canvas-rendered
// typography, not DOM text" — see components/canvas/TaglineParticles.jsx,
// which lives in the Canvas, not here. This file exists as an explicit
// placeholder (rather than simply omitting Act 2 from PinnedActs.jsx) so
// the five-act structure stays visible in the component tree, and as the
// obvious place to add a small supporting DOM caption later if the
// design calls for one alongside the particle typography.
export default function Act2EventHorizon() {
  return null;
}
