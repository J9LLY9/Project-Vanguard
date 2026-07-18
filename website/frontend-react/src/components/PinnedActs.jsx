import { useEffect, useRef, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import Experience from "./canvas/Experience";
import Act1Singularity from "./acts/Act1Singularity";
import Act2EventHorizon from "./acts/Act2EventHorizon";
import Act3Supernova from "./acts/Act3Supernova";
import MobileHeroFallback from "./mobile/MobileHeroFallback";
import { useScrollTimeline } from "../hooks/useScrollTimeline";
import { probeInitialTier } from "../lib/fpsTier";
import { isMobileDevice } from "../lib/deviceCapabilities";
import { useSceneStore } from "../state/sceneStore";

/**
 * Owns the ONE pinned scroll region covering Acts 1-3 — the `<section>`
 * below is what hooks/useScrollTimeline.js actually pins — plus the ONE
 * <Canvas> all three acts' 3D content shares (components/canvas/Experience.jsx).
 * Native scroll resumes immediately after this section for Act 4 onward
 * (see App.jsx), with no extra wiring required beyond this section simply
 * ending.
 *
 * Mobile gets a completely separate, lighter-weight branch
 * (components/mobile/MobileHeroFallback.jsx) rather than the desktop
 * particle pipeline at a reduced count — see that file for why.
 */
export default function PinnedActs() {
  const pinTargetRef = useRef();
  const [ready, setReady] = useState(false);
  const isMobile = useSceneStore((s) => s.isMobile);
  const particleConfig = useSceneStore((s) => s.particleConfig);
  const setQuality = useSceneStore((s) => s.setQuality);
  const setIsMobile = useSceneStore((s) => s.setIsMobile);

  useEffect(() => {
    const onMobile = isMobileDevice();
    setIsMobile(onMobile);

    if (onMobile) {
      // Skip the desktop FPS probe entirely for the mobile branch — it
      // never mounts the GPGPU particle pipeline the probe exists to
      // size, so there's nothing for it to inform.
      setReady(true);
      return;
    }

    probeInitialTier().then(({ tier, config }) => {
      setQuality(tier, config);
      setReady(true);
    });
  }, [setQuality, setIsMobile]);

  // Always called (rules of hooks) — internally becomes a no-op-ish
  // unpinned tracker under prefers-reduced-motion; see that file.
  useScrollTimeline(pinTargetRef);

  // The pin is created (and ScrollTrigger.refresh()'d) while `ready` is
  // still false, against whatever the loading placeholder measures at —
  // once the Canvas/Acts mount in, re-measure in case anything below
  // this section on the page depends on its final layout.
  useEffect(() => {
    if (ready) ScrollTrigger.refresh();
  }, [ready]);

  return (
    <section ref={pinTargetRef} className="pinned-acts">
      {!ready ? (
        <div className="pinned-acts__loading" aria-hidden="true" />
      ) : isMobile ? (
        <MobileHeroFallback />
      ) : (
        <>
          <Canvas
            className="pinned-acts__canvas"
            gl={{ antialias: true, powerPreference: "high-performance" }}
            dpr={[1, 2]}
          >
            <Experience particleConfig={particleConfig} />
          </Canvas>
          <Act1Singularity />
          <Act2EventHorizon />
          <Act3Supernova />
        </>
      )}
    </section>
  );
}
