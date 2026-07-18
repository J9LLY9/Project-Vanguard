import { useEffect, useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { PALETTE } from "../../lib/constants";

gsap.registerPlugin(ScrollTrigger);

const SHAPES = {
  icosahedron: () => new THREE.IcosahedronGeometry(1.2, 0),
  box: () => new THREE.BoxGeometry(1.6, 1.6, 1.6),
  torus: () => new THREE.TorusGeometry(1.1, 0.4, 16, 48),
};

function RevealMesh({ shape, progressRef, accentColor }) {
  const geometry = useMemo(() => (SHAPES[shape] || SHAPES.icosahedron)(), [shape]);
  const wireMaterial = useMemo(
    () => new THREE.MeshBasicMaterial({ color: accentColor, wireframe: true, transparent: true }),
    [accentColor]
  );
  const solidMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: accentColor,
        roughness: 0.35,
        metalness: 0.1,
        transparent: true,
        opacity: 0,
      }),
    [accentColor]
  );
  const groupRef = useRef();

  useFrame((_, delta) => {
    const t = progressRef.current;
    wireMaterial.opacity = 1 - t;
    solidMaterial.opacity = t;
    if (groupRef.current) groupRef.current.rotation.y += delta * 0.2;
  });

  return (
    <group ref={groupRef}>
      <mesh geometry={geometry} material={wireMaterial} />
      <mesh geometry={geometry} material={solidMaterial} />
    </group>
  );
}

/**
 * The recurring "wireframe-to-solid" reveal used across Act 4's feature
 * sections: a small embedded Canvas whose shape crossfades from
 * wireframe to a lit solid as its DOM wrapper scrolls into view. Uses
 * its OWN scoped ScrollTrigger (start/end tied to ITS OWN wrapper
 * element's viewport position, `scrub: true`) rather than the Acts 1-3
 * pin's timeline — Act 4 is native scroll, so each instance of this
 * component reveals independently as the user scrolls past it, not in
 * lockstep with a master pinned timeline.
 */
export default function WireframeReveal({ shape = "icosahedron", accentColor = PALETTE.teal, className = "" }) {
  const wrapperRef = useRef();
  const progressRef = useRef(0);

  useEffect(() => {
    if (!wrapperRef.current) return undefined;
    const trigger = ScrollTrigger.create({
      trigger: wrapperRef.current,
      start: "top 80%",
      end: "top 35%",
      scrub: true,
      onUpdate: (self) => {
        progressRef.current = self.progress;
      },
    });
    return () => trigger.kill();
  }, []);

  return (
    <div ref={wrapperRef} className={`wireframe-reveal ${className}`}>
      <Canvas camera={{ position: [0, 0, 4], fov: 40 }} dpr={[1, 1.75]}>
        <ambientLight intensity={0.6} />
        <directionalLight position={[3, 3, 4]} intensity={1.2} />
        <RevealMesh shape={shape} progressRef={progressRef} accentColor={accentColor} />
      </Canvas>
    </div>
  );
}
