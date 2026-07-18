import { PARTICLE_TIERS } from "./constants";
import { getHardwareHint } from "./deviceCapabilities";

// Probe the device for ~1s BEFORE the heavy GPGPU particle system ever
// mounts, and pick a particle-count tier from it. This has to run
// up front and pick a tier ONCE — the GPGPU simulation allocates
// fixed-size ping-pong render targets sized to a particle count (see
// gpgpu/useDustSimulation.js), so the count can't cheaply change after
// the sim is created without tearing the whole thing down and rebuilding
// it. `@react-three/drei`'s <PerformanceMonitor> (wired in
// components/canvas/QualityController.jsx) handles ONGOING adaptation
// after mount — dropping DPR/post-processing quality — but not particle
// COUNT, for the same reason.
//
// The probe itself: rather than spinning up a real Three.js scene just
// to measure it (expensive, and the thing we're trying to avoid doing
// twice), we sample requestAnimationFrame deltas for a fixed window.
// That measures the browser/compositor's baseline frame pacing under
// whatever's already on the page (React mounting, layout, etc.) — a
// reasonable proxy for "how much headroom does this device have," and
// zero-cost to combine with a hardware-concurrency hint as a tie-breaker.
const PROBE_DURATION_MS = 1000;

function sampleFrameRate(durationMs) {
  return new Promise((resolve) => {
    const samples = [];
    let last = performance.now();

    function tick(now) {
      samples.push(now - last);
      last = now;
      if (now - start < durationMs) {
        requestAnimationFrame(tick);
      } else {
        resolve(samples);
      }
    }

    const start = performance.now();
    requestAnimationFrame(tick);
  });
}

function averageFps(deltas) {
  if (!deltas.length) return 60;
  const meanDelta = deltas.reduce((sum, d) => sum + d, 0) / deltas.length;
  return meanDelta > 0 ? 1000 / meanDelta : 60;
}

/**
 * @returns {Promise<{tier: keyof typeof PARTICLE_TIERS, fps: number}>}
 */
export async function probeInitialTier() {
  const deltas = await sampleFrameRate(PROBE_DURATION_MS);
  const fps = averageFps(deltas);
  const { cores } = getHardwareHint();

  let tier;
  if (fps >= 55 && cores >= 6) {
    tier = "high";
  } else if (fps >= 45 && cores >= 4) {
    tier = "mid";
  } else if (fps >= 24) {
    tier = "low";
  } else {
    tier = "mobile";
  }

  return { tier, fps, config: PARTICLE_TIERS[tier] };
}
