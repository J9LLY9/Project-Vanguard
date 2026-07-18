import { create } from "zustand";

// Shared state between the GSAP scroll timeline (which lives outside
// React, see hooks/useScrollTimeline.js), the r3f Canvas (which reads
// scroll progress every frame to drive shader uniforms), and plain DOM
// components (Nav, ScrollProgress) that need the same numbers without
// subscribing to Three.js's render loop.
//
// Split into two progress values on purpose:
//   - `pinnedProgress`  0..1 across the PINNED Acts 1-3 region only.
//   - `pageProgress`    0..1 across the ENTIRE page (all five acts) —
//                        this is what ScrollProgress.jsx renders, since
//                        the indicator has to span all five acts, not
//                        just the pinned/scroll-jacked portion.
// Components inside the pin (particles, sphere, flash) read
// `pinnedProgress`; Nav/ScrollProgress read `pageProgress`.
export const useSceneStore = create((set) => ({
  pinnedProgress: 0,
  pageProgress: 0,
  pinComplete: false, // true once Act 3 has fully resolved — gates the Nav fade-in
  qualityTier: "mid", // overwritten by the fpsTier probe before Canvas mounts
  particleConfig: null, // { dust, textureSize } for the resolved tier
  isMobile: false,

  setPinnedProgress: (v) => set({ pinnedProgress: v }),
  setPageProgress: (v) => set({ pageProgress: v }),
  setPinComplete: (v) => set({ pinComplete: v }),
  setQuality: (tier, particleConfig) => set({ qualityTier: tier, particleConfig }),
  setIsMobile: (v) => set({ isMobile: v }),
}));
