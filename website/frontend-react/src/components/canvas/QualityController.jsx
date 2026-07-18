import { useThree } from "@react-three/fiber";
import { PerformanceMonitor } from "@react-three/drei";

/**
 * Ongoing quality adaptation AFTER mount — complements, doesn't replace,
 * lib/fpsTier.js's one-time PRE-mount probe (which fixes the particle
 * COUNT for the GPGPU sim's lifetime; see that file for why count can't
 * cheaply change afterward). This only ever adjusts device pixel ratio:
 * drei's <PerformanceMonitor> samples real sustained frame times and
 * fires onDecline/onIncline as performance actually drifts (thermal
 * throttling, another app stealing GPU time, a background tab
 * resuming) — the cheapest lever available without touching the
 * particle system itself.
 */
export default function QualityController() {
  const { gl } = useThree();

  return (
    <PerformanceMonitor
      onDecline={() => gl.setPixelRatio(Math.max(1, gl.getPixelRatio() - 0.25))}
      onIncline={() => gl.setPixelRatio(Math.min(2, gl.getPixelRatio() + 0.25))}
      onFallback={() => gl.setPixelRatio(1)}
      flipflops={3}
    />
  );
}
