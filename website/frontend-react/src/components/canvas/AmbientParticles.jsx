import { useMemo, useRef } from "react";
import * as THREE from "three";
import { useFrame } from "@react-three/fiber";
import { PALETTE } from "../../lib/constants";
import vertexShader from "../../shaders/ambient/ambientDrift.vert.glsl?raw";
import fragmentShader from "../../shaders/ambient/ambientDrift.frag.glsl?raw";

const AMBIENT_COUNT = 800;

/**
 * Act 4's "ambient particle motif ... low density/no physics, calm
 * drift only" — deliberately the simplest particle system in this
 * project: plain THREE.Points with a cheap sine drift in the vertex
 * shader (still zero per-frame JS work), no GPGPU simulation, no
 * gravity, no anisotropic stretch. Act 4 is meant to feel settled after
 * Acts 1-3's collapse; reusing the dust cloud's heavier machinery here
 * would cost more than this section needs and would read as visually
 * inconsistent with "calm."
 */
export default function AmbientParticles() {
  const pointsRef = useRef();

  const geometry = useMemo(() => {
    const positions = new Float32Array(AMBIENT_COUNT * 3);
    const seeds = new Float32Array(AMBIENT_COUNT);
    for (let i = 0; i < AMBIENT_COUNT; i++) {
      positions[i * 3] = (Math.random() - 0.5) * 30;
      positions[i * 3 + 1] = (Math.random() - 0.5) * 30;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 10;
      seeds[i] = Math.random();
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setAttribute("aSeed", new THREE.BufferAttribute(seeds, 1));
    return geo;
  }, []);

  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader,
        fragmentShader,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        uniforms: {
          uTime: { value: 0 },
          uColor: { value: new THREE.Color(PALETTE.teal) },
        },
      }),
    []
  );

  useFrame((_, delta) => {
    material.uniforms.uTime.value += delta;
  });

  return <points ref={pointsRef} geometry={geometry} material={material} frustumCulled={false} />;
}
