import { useMemo, useRef } from "react";
import * as THREE from "three";
import { useFrame } from "@react-three/fiber";
import { timelineUniforms } from "../../hooks/useScrollTimeline";
import { PALETTE } from "../../lib/constants";
import vertexShader from "../../shaders/supernova/flash.vert.glsl?raw";
import fragmentShader from "../../shaders/supernova/flash.frag.glsl?raw";

/**
 * Act 3's white burst: a single large camera-facing plane whose shader
 * shapes its own "bursts, peaks near-solid white, then clears" envelope
 * from `uFlash` alone — see flash.frag.glsl. Shader-driven, not a video
 * asset. Sized generously and placed at the origin so it fills the
 * frame from any reasonable camera distance in THIS scene without
 * tracking the camera's frustum — Acts 1-3 don't move the camera. A
 * scene with real camera movement through this act would need to either
 * re-size this plane per frame or parent it to the camera instead.
 */
export default function SupernovaFlash() {
  const meshRef = useRef();

  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader,
        fragmentShader,
        transparent: true,
        depthWrite: false,
        depthTest: false,
        uniforms: {
          uFlash: { value: 0 },
          uTime: { value: 0 },
          uColorTeal: { value: new THREE.Color(PALETTE.tealBright) },
          uColorWhite: { value: new THREE.Color(PALETTE.white) },
        },
      }),
    []
  );

  useFrame((_, delta) => {
    material.uniforms.uFlash.value = timelineUniforms.flash;
    material.uniforms.uTime.value += delta;
  });

  return (
    <mesh ref={meshRef} material={material} renderOrder={10}>
      <planeGeometry args={[40, 40]} />
    </mesh>
  );
}
