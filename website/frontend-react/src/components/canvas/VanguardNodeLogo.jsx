import { useMemo, useRef } from "react";
import * as THREE from "three";
import { useFrame } from "@react-three/fiber";
import { timelineUniforms } from "../../hooks/useScrollTimeline";
import { PALETTE } from "../../lib/constants";

/**
 * The reveal target of Act 3: a plain wireframe icosahedron. Deliberately
 * NOT built from particles and NOT eased/assembled into view — the brief
 * is explicit that this should "land instantly and still," so its ONLY
 * per-frame behavior is a slow idle spin (reads as "alive," not
 * "assembling") plus a hard visibility flip keyed to the flash's clear
 * point. Nothing about its appearance animates IN.
 */
export default function VanguardNodeLogo() {
  const groupRef = useRef();

  const geometry = useMemo(() => new THREE.IcosahedronGeometry(1.1, 1), []);
  const material = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        color: new THREE.Color(PALETTE.teal),
        wireframe: true,
        transparent: true,
        opacity: 0.9,
      }),
    []
  );

  useFrame((_, delta) => {
    const group = groupRef.current;
    if (!group) return;

    // Hard cut at the flash's midpoint, not a fade/scale-in — everything
    // ELSE in this scene eases; this deliberately does not.
    group.visible = timelineUniforms.flash > 0.5;
    group.rotation.y += delta * 0.08;
    group.rotation.x += delta * 0.03;
  });

  return (
    <group ref={groupRef} visible={false}>
      <mesh geometry={geometry} material={material} />
    </group>
  );
}
