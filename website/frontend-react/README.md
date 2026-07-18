# Vanguard — "The Neural Dive" landing page

A cinematic, scroll-driven landing page scaffold for Vanguard, built with
React + `@react-three/fiber` (Three.js) + GSAP `ScrollTrigger`, on Vite.

## Running it

```bash
npm install
npm run dev
```

This project was scaffolded in an environment without a Node.js runtime
available, so **`npm install`/`npm run dev`/`npm run build` have not been
run or verified here.** Everything was checked by hand instead: every
relative import resolves to a real file, every named/default export used
actually exists, every shader `uniform`/`attribute` name is matched
exactly by the JS component that sets it, and every external package
imported is declared in `package.json`. That's a strong signal the
project is internally consistent, but it is **not** a substitute for an
actual `npm install && npm run dev` — please do that first and treat the
first console/browser errors as more authoritative than anything in this
README.

## Structure — five acts, one continuous page

```
src/
  components/
    PinnedActs.jsx        # owns the scroll-jacked pin covering Acts 1-3, and the ONE shared <Canvas>
    canvas/                # everything that renders INSIDE that Canvas
      Experience.jsx       # root orchestrator for Acts 1-3's 3D content
      DustSystem.jsx        # owns the GPGPU sim, shared by NeuralDust + DataLines
      NeuralDust.jsx         # the 10-20k particle "neural dust" cloud
      DataLines.jsx          # faint connections between nearby particles
      SingularitySphere.jsx # Act 1's gravitational-lens sphere
      TaglineParticles.jsx  # Act 2's tagline, sampled from canvas typography
      SupernovaFlash.jsx    # Act 3's white burst
      VanguardNodeLogo.jsx  # Act 3's reveal target
      AmbientParticles.jsx  # Act 4's lightweight, no-physics particle motif
      QualityController.jsx # ongoing DPR adaptation (PerformanceMonitor)
    acts/
      Act1Singularity.jsx / Act2EventHorizon.jsx / Act3Supernova.jsx   # DOM overlay captions for the pinned acts
      Act4NewWorld.jsx      # ACT 4 — flagged in-file as needing the most build-out; see its TODOs
      act4/OptimizationProof.jsx, act4/FeatureSection.jsx
      Act5Invitation.jsx    # CTA / signup / footer
    layout/
      Nav.jsx               # fades in only once Act 3 completes
      ScrollProgress.jsx    # spans all five acts
      WireframeReveal.jsx   # the recurring wireframe-to-solid feature transition
    mobile/
      MobileHeroFallback.jsx # looped-video branch for Acts 1-3 on mobile
  gpgpu/
    useDustSimulation.js   # GPUComputationRenderer ping-pong FBO wrapper
  shaders/                 # every shader as its own .glsl file, imported with Vite's `?raw`
  hooks/useScrollTimeline.js # the GSAP ScrollTrigger timeline (the pin + the page-wide progress tracker)
  lib/                      # constants, FPS-tier probe, device checks, typography sampling, seeded particle distribution
  state/sceneStore.js       # zustand store shared between the Canvas and plain DOM components
```

## What's real vs. what's a placeholder

**Real and functional:** the GPGPU ping-pong simulation and its
gravitational-attraction physics; the `gl_InstanceID`-driven instanced
particle rendering with anisotropic (velocity-aligned) stretch; the
render-to-texture gravitational-lens sphere; the canvas-sampled
typography particle field; the shader-driven supernova flash; the
scroll-jack pin constrained to Acts 1-3 with native scroll resuming
after; the FPS-tier probe and its particle-count fallback; the
`prefers-reduced-motion`/mobile branches; the recurring
wireframe-to-solid reveal; the scroll-progress indicator.

**Explicitly placeholder (see in-file `TODO` comments):**

- `components/acts/Act4NewWorld.jsx` and its children — real product
  copy, a deliberate icon/shape choice per feature, real benchmark
  numbers in `OptimizationProof.jsx`.
- `components/acts/Act5Invitation.jsx` — CTA copy and the actual
  signup/download destination (the button is currently inert).
- `components/mobile/MobileHeroFallback.jsx` — `VIDEO_SRC` points at a
  placeholder path; it needs a real exported rendering of the Acts 1-3
  sequence once that sequence's look is locked on desktop.

## Design decisions worth knowing before you extend this

- **The dust simulation is owned once, shared twice.** `DustSystem.jsx`
  is the only thing that calls `gpu.step()`; `NeuralDust.jsx` and
  `DataLines.jsx` only read the resulting textures. Don't add a second
  `useDustSimulation()` call anywhere — see that hook's and
  `DustSystem.jsx`'s comments for why sharing the one instance matters
  (both for correctness — two independent sims would drift apart
  immediately — and for GPU cost).
- **`timelineUniforms` (in `hooks/useScrollTimeline.js`) is a plain
  mutable object, not React/zustand state**, specifically so the Canvas
  can read scroll-driven values every frame without triggering React
  re-renders at 60fps. DOM components that need the same numbers
  reactively (`Nav`, `ScrollProgress`, the act captions) read from the
  zustand store instead, which the same GSAP `onUpdate` also writes to.
- **Particle count is fixed for the GPGPU sim's lifetime** once
  `lib/fpsTier.js`'s probe resolves — see that file for why it can't
  cheaply change after mount. `QualityController.jsx`'s ongoing
  adaptation only ever touches device pixel ratio, not particle count.
- **No custom `useFrame` priorities are used anywhere.** Passing a
  non-default priority to any `useFrame` in an r3f tree switches off
  automatic rendering for the whole Canvas, and something then has to
  call `gl.render()` manually. This project avoids that entirely by
  relying on React's own parent-before-child, earlier-sibling-before-later
  render order instead — see `SingularitySphere.jsx`'s and
  `DustSystem.jsx`'s comments. Keep that in mind before reaching for
  `useFrame(cb, priority)` elsewhere in this codebase.
