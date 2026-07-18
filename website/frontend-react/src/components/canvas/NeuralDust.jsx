import { useMemo, useRef } from "react";
import * as THREE from "three";
import { useFrame } from "@react-three/fiber";
import { timelineUniforms } from "../../hooks/useScrollTimeline";
import { PALETTE } from "../../lib/constants";
import vertexShader from "../../shaders/particles/neuralDust.vert.glsl?raw";
import fragmentShader from "../../shaders/particles/neuralDust.frag.glsl?raw";

/**
 * Teacher Mode — why RawShaderMaterial + GLSL3, not a plain
 * ShaderMaterial: this shader reads `gl_InstanceID` to look up each
 * particle's position/velocity straight out of the GPGPU simulation
 * textures (see gpgpu/useDustSimulation.js, owned by the parent
 * DustSystem.jsx and passed in as `gpu`) instead of via InstancedMesh's
 * usual per-instance `instanceMatrix` — which is also why `setMatrixAt`
 * is never called anywhere in this file; every instance's true transform
 * is derived on the GPU from the simulation, every frame.
 * `gl_InstanceID` is a GLSL ES 3.00 built-in, unavailable in the GLSL ES
 * 1.00 dialect a plain `ShaderMaterial` compiles to. `RawShaderMaterial`
 * with `glslVersion: THREE.GLSL3` gives full manual control of the
 * shader text (no auto-injected `attribute`/`varying` declarations that
 * would collide with GLSL3's `in`/`out` syntax) while Three.js still
 * transparently uploads the standard `modelViewMatrix`/`projectionMatrix`
 * uniforms every frame — it binds those by NAME against whatever the
 * compiled program actually declares, regardless of Raw vs. non-Raw, so
 * "Raw" only means "no auto-injected shader TEXT," not "no automatic
 * per-object uniforms."
 *
 * This component does NOT call `gpu.step()` — see DustSystem.jsx, which
 * owns the single per-frame compute call shared with DataLines.jsx. This
 * component only READS the resulting textures. There is no loop over
 * particles anywhere in this file — the entire per-frame JS cost is two
 * texture-reference assignments plus two scalar uniform writes.
 */
export default function NeuralDust({ gpu, count, textureSize }) {
  const geometry = useMemo(() => new THREE.PlaneGeometry(1, 1), []);

  const material = useMemo(
    () =>
      new THREE.RawShaderMaterial({
        glslVersion: THREE.GLSL3,
        vertexShader,
        fragmentShader,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        uniforms: {
          uPositionTexture: { value: null },
          uVelocityTexture: { value: null },
          uTextureSize: { value: textureSize },
          uBaseSize: { value: 0.09 },
          uStretchFactor: { value: 9 },
          uMaxStretch: { value: 12 },
          uGravity: { value: 0 },
          uFlash: { value: 0 },
          uColorTeal: { value: new THREE.Color(PALETTE.tealBright) },
          uColorObsidian: { value: new THREE.Color(PALETTE.obsidian) },
          uColorWhite: { value: new THREE.Color(PALETTE.white) },
        },
      }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [textureSize]
  );

  const meshRef = useRef();

  useFrame(() => {
    material.uniforms.uPositionTexture.value = gpu.getPositionTexture();
    material.uniforms.uVelocityTexture.value = gpu.getVelocityTexture();
    material.uniforms.uGravity.value = timelineUniforms.gravity;
    material.uniforms.uFlash.value = timelineUniforms.flash;
  });

  return <instancedMesh ref={meshRef} args={[geometry, material, count]} frustumCulled={false} />;
}
