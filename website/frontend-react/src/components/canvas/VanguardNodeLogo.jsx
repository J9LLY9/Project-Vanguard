import { useMemo, useRef } from "react";
import * as THREE from "three";
import { Canvas, useFrame } from "@react-three/fiber";
import { PALETTE } from "../../lib/constants";

// Idle rotation + a subtle breathing opacity pulse — purely ambient,
// driven off elapsed time only. No scroll coupling of any kind.
function NodeMesh() {
  const groupRef = useRef();
  const innerMaterial = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        color: new THREE.Color(PALETTE.accent),
        wireframe: true,
        transparent: true,
        opacity: 0.9,
      }),
    []
  );
  const outerMaterial = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        color: new THREE.Color(PALETTE.accentDim),
        wireframe: true,
        transparent: true,
        opacity: 0.25,
      }),
    []
  );
  const geometry = useMemo(() => new THREE.IcosahedronGeometry(1.15, 1), []);
  const outerGeometry = useMemo(() => new THREE.IcosahedronGeometry(1.55, 0), []);

  useFrame((state, delta) => {
    const group = groupRef.current;
    if (!group) return;
    group.rotation.y += delta * 0.12;
    group.rotation.x += delta * 0.045;
    const pulse = Math.sin(state.clock.elapsedTime * 0.6) * 0.5 + 0.5;
    innerMaterial.opacity = 0.7 + pulse * 0.25;
    outerMaterial.opacity = 0.12 + pulse * 0.18;
  });

  return (
    <group ref={groupRef}>
      <mesh geometry={geometry} material={innerMaterial} />
      <mesh geometry={outerGeometry} material={outerMaterial} />
    </group>
  );
}

/**
 * The hero's node mark: a small idle-rotating wireframe icosahedron with
 * a soft pulse, replacing the old singularity/sphere reveal. Owns its
 * own Canvas — the hero just drops this in and lays it out.
 */
export default function VanguardNodeLogo() {
  return (
    <div className="hero__node" aria-hidden="true">
      <div className="hero__node-glow" />
      <Canvas camera={{ position: [0, 0, 4.2], fov: 40 }} dpr={[1, 2]} gl={{ antialias: true, alpha: true }}>
        <NodeMesh />
      </Canvas>
    </div>
  );
}
