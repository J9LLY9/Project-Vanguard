import { useEffect } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { useSceneStore } from "../state/sceneStore";
import { ACT_BOUNDS } from "../lib/constants";
import { prefersReducedMotion, isMobileDevice } from "../lib/deviceCapabilities";

gsap.registerPlugin(ScrollTrigger);

// Mutable singleton, NOT React state. The r3f Canvas reads these every
// `useFrame` (60x/sec); re-rendering the whole React tree at that rate
// just to push a scroll number into a shader uniform would reintroduce,
// in React form, exactly the "per-frame JS work" the brief asks us to
// avoid in the particle systems. GSAP's onUpdate (scroll-tick frequency,
// not every rAF) writes here; each canvas component reads the CURRENT
// value inside its own useFrame — no subscription, no re-render.
export const timelineUniforms = {
  pinnedProgress: 0,
  gravity: 0, // Act 2's "gravitational constant" — 0..1
  flash: 0, // Act 3's supernova progress — 0..1
};

function remap(value, start, end) {
  if (end === start) return 0;
  return Math.min(1, Math.max(0, (value - start) / (end - start)));
}

function smoothstep(t) {
  const c = Math.min(1, Math.max(0, t));
  return c * c * (3 - 2 * c);
}

/**
 * Two independent GSAP ScrollTriggers:
 *
 * 1. THE PIN — scroll-jacks Acts 1-3 for a tall runway (`end: "+=400%"`,
 *    i.e. 4 viewport-heights of scroll distance), scrubbed, and derives
 *    `gravity`/`flash` from the SAME `ACT_BOUNDS` used everywhere else
 *    (lib/constants.js) so the physics, the shaders, and any DOM
 *    captions can never drift out of sync with each other. This is the
 *    ONLY pinned trigger on the page — scroll-jacking is constrained to
 *    exactly this region, and native scroll resumes for Act 4 onward
 *    the instant its `end` is passed, with no extra wiring required.
 *
 * 2. THE PAGE-WIDE PROGRESS TRACKER — unpinned, spans the entire
 *    document, feeds `pageProgress` into the zustand store for
 *    ScrollProgress.jsx, which has to span all FIVE acts, not just the
 *    pinned portion — kept deliberately separate from the pin's own
 *    progress so the indicator doesn't freeze once the pin releases.
 *
 * `prefers-reduced-motion` disables `pin: true` specifically — forced
 * scroll-jacking (the page hijacking how far one scroll gesture moves
 * you) is exactly the kind of motion that preference exists to opt out
 * of. Mobile skips pinning for the same UX reason (scroll-jacking on a
 * touch device reads as broken/unresponsive far more often than it reads
 * as cinematic) — though mobile mounts MobileHeroFallback.jsx instead of
 * this Canvas anyway, so this mostly matters for the edge case of a
 * mobile-sized viewport that DIDN'T trip `isMobileDevice()`'s signals
 * (see lib/deviceCapabilities.js). Either way, the gravity/flash physics
 * and shader transitions themselves stay driven by scroll position
 * (turning those off entirely would leave Acts 1-3 permanently stuck at
 * their rest state, which isn't what either condition is asking for) —
 * only the PINNING is conditional; unpinned, the same transform simply
 * plays out over the section's natural height as the user scrolls past
 * it normally.
 */
export function useScrollTimeline(pinTargetRef) {
  const setPinnedProgress = useSceneStore((s) => s.setPinnedProgress);
  const setPageProgress = useSceneStore((s) => s.setPageProgress);
  const setPinComplete = useSceneStore((s) => s.setPinComplete);

  useEffect(() => {
    if (!pinTargetRef.current) return undefined;

    const shouldPin = !prefersReducedMotion() && !isMobileDevice();

    const pinTrigger = ScrollTrigger.create({
      trigger: pinTargetRef.current,
      start: "top top",
      end: "+=400%",
      pin: shouldPin,
      scrub: 1,
      anticipatePin: 1,
      onUpdate: (self) => {
        const progress = self.progress;
        timelineUniforms.pinnedProgress = progress;

        const gravityT = remap(progress, ACT_BOUNDS.eventHorizon.start, ACT_BOUNDS.eventHorizon.end);
        timelineUniforms.gravity = smoothstep(gravityT);

        // flash.frag.glsl shapes its own burst-then-clear curve from
        // this raw 0..1 ramp — see that file for the envelope.
        timelineUniforms.flash = remap(progress, ACT_BOUNDS.supernova.start, ACT_BOUNDS.supernova.end);

        setPinnedProgress(progress);
        setPinComplete(progress >= 0.999);
      },
    });

    const pageTrigger = ScrollTrigger.create({
      trigger: document.body,
      start: "top top",
      end: "bottom bottom",
      scrub: 0.3,
      onUpdate: (self) => setPageProgress(self.progress),
    });

    ScrollTrigger.refresh();

    return () => {
      pinTrigger.kill();
      pageTrigger.kill();
    };
  }, [pinTargetRef, setPinnedProgress, setPageProgress, setPinComplete]);
}
