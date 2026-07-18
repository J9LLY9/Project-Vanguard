import { useMemo, useRef } from "react";
import * as THREE from "three";
import { useFrame } from "@react-three/fiber";
import { sampleTextToParticles } from "../../lib/typographyParticles";
import { timelineUniforms } from "../../hooks/useScrollTimeline";
import { PALETTE, TAGLINE_PARTICLE_COUNT, TAGLINE_TEXT } from "../../lib/constants";
import vertexShader from "../../shaders/typography/typographyParticles.vert.glsl?raw";
import fragmentShader from "../../shaders/typography/typographyParticles.frag.glsl?raw";

/**
 * Act 2's tagline, "rendered as a particle field sampled from
 * canvas-rendered typography, not DOM text" — see
 * lib/typographyParticles.js for the canvas-sampling step, run ONCE on
 * mount via useMemo. Reuses the dust cloud's GPU-instanced-quad +
 * anisotropic-stretch-toward-center technique, but WITHOUT a second
 * GPGPU simulation — see typographyParticles.vert.glsl's header comment
 * for why an analytic pull is the right-sized tool here instead.
 */
export default function TaglineParticles() {
  const meshRef = useRef();

  const { geometry, count } = useMemo(() => {
    const basePositions = sampleTextToParticles(TAGLINE_TEXT, TAGLINE_PARTICLE_COUNT, 9);
    const seeds = new Float32Array(TAGLINE_PARTICLE_COUNT);
    for (let i = 0; i < TAGLINE_PARTICLE_COUNT; i++) seeds[i] = Math.random();

    const geo = new THREE.PlaneGeometry(1, 1);
    geo.setAttribute("aBasePosition", new THREE.InstancedBufferAttribute(basePositions, 3));
    geo.setAttribute("aSeed", new THREE.InstancedBufferAttribute(seeds, 1));

    return { geometry: geo, count: TAGLINE_PARTICLE_COUNT };
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
          uGravity: { value: 0 },
          uFlash: { value: 0 },
          uBaseSize: { value: 0.055 },
          uColorTeal: { value: new THREE.Color(PALETTE.tealBright) },
          uColorWhite: { value: new THREE.Color(PALETTE.white) },
        },
      }),
    []
  );

  useFrame(() => {
    material.uniforms.uGravity.value = timelineUniforms.gravity;
    material.uniforms.uFlash.value = timelineUniforms.flash;
  });

  return <instancedMesh ref={meshRef} args={[geometry, material, count]} frustumCulled={false} />;
}
