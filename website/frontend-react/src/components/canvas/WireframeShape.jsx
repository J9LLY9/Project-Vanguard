import { useMemo, useRef } from "react";
import * as THREE from "three";
import { Canvas, useFrame } from "@react-three/fiber";
import { PALETTE } from "../../lib/constants";

const SHAPES = {
  icosahedron: () => new THREE.IcosahedronGeometry(1.2, 0),
  cube: () => new THREE.BoxGeometry(1.5, 1.5, 1.5),
  torus: () => new THREE.TorusGeometry(1.05, 0.38, 16, 48),
};

function ShapeMesh({ shape, accentColor }) {
  const groupRef = useRef();
  const geometry = useMemo(() => (SHAPES[shape] || SHAPES.icosahedron)(), [shape]);
  const material = useMemo(
    () => new THREE.MeshBasicMaterial({ color: accentColor, wireframe: true }),
    [accentColor]
  );

  useFrame((_, delta) => {
    if (!groupRef.current) return;
    groupRef.current.rotation.y += delta * 0.18;
    groupRef.current.rotation.x += delta * 0.06;
  });

  return (
    <group ref={groupRef}>
      <mesh geometry={geometry} material={material} />
    </group>
  );
}

/**
 * Small idle-rotating wireframe shape used in the feature grid — ambient
 * motion only, no wireframe-to-solid crossfade and no per-instance
 * scroll trigger. The grid's own entrance fade (see useInView) is what
 * brings this into view; once mounted it just spins forever.
 */
export default function WireframeShape({ shape = "icosahedron", accentColor = PALETTE.accent, className = "" }) {
  return (
    <div className={`wireframe-shape ${className}`} aria-hidden="true">
      <Canvas camera={{ position: [0, 0, 4], fov: 40 }} dpr={[1, 1.75]} gl={{ antialias: true, alpha: true }}>
        <ShapeMesh shape={shape} accentColor={accentColor} />
      </Canvas>
    </div>
  );
}
