// Cheap, synchronous device checks — no probing/timing here (see
// fpsTier.js for the async FPS probe). Used to pick the mobile branch
// before we even mount the desktop Canvas.

export function isMobileDevice() {
  if (typeof window === "undefined") return false;
  const coarsePointer = window.matchMedia?.("(pointer: coarse)").matches ?? false;
  const narrowViewport = window.innerWidth < 820;
  const uaMobile = /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent || "");
  // Require two of three signals so a coarse-pointer laptop touchscreen
  // (still plenty powerful) doesn't get routed onto the lightweight path.
  const signals = [coarsePointer, narrowViewport, uaMobile].filter(Boolean).length;
  return signals >= 2;
}

export function prefersReducedMotion() {
  if (typeof window === "undefined") return false;
  return window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
}

export function getHardwareHint() {
  // Rough, best-effort signal only — combined with the real FPS probe in
  // fpsTier.js, never used alone to pick a tier.
  return {
    cores: navigator.hardwareConcurrency || 4,
    memoryGB: navigator.deviceMemory || 4,
  };
}
