import { useMemo, useRef } from "react";
import * as THREE from "three";
import { useFrame, useThree } from "@react-three/fiber";
import { useFBO } from "@react-three/drei";
import { timelineUniforms } from "../../hooks/useScrollTimeline";
import { PALETTE } from "../../lib/constants";
import vertexShader from "../../shaders/singularity/lensSphere.vert.glsl?raw";
import fragmentShaderBody from "../../shaders/singularity/lensSphere.frag.glsl?raw";
import noiseGLSL from "../../shaders/common/noise.glsl?raw";

const fragmentShader = `${noiseGLSL}\n${fragmentShaderBody}`;

/**
 * Teacher Mode — the "gravitational lens": a genuine render-to-texture
 * pass, not an environment-map approximation. Every frame, BEFORE the
 * normal render: hide this sphere, render the scene into an offscreen
 * `useFBO` target (so the FBO ends up holding "everything except the
 * lens itself," dust cloud included, at its CURRENT frame position),
 * restore the sphere's visibility, and hand that texture to the
 * sphere's own shader as `uBackdrop`. The shader then samples it with a
 * UV offset bent toward screen center — see lensSphere.frag.glsl — so
 * whatever's behind the sphere visibly warps inward, strongest at the
 * silhouette and growing with `uGravity`.
 *
 * This does NOT use a custom `useFrame` priority, and specifically does
 * NOT call `gl.render()` a second time for a "final" pass — the extra
 * `gl.render(scene, camera)` call below targets the OFFSCREEN `fbo`
 * only; react-three-fiber's own automatic per-frame render (unaffected,
 * since nothing in this Canvas uses a non-default priority — see
 * DustSystem.jsx) still runs afterward and produces the actual on-screen
 * frame, now showing the sphere with its freshly captured backdrop.
 * The only ordering requirement — capture the backdrop AFTER the dust
 * simulation has advanced for this frame — is satisfied by mounting
 * `<SingularitySphere />` after `<DustSystem />` in Experience.jsx,
 * since React registers parent/earlier-sibling `useFrame` callbacks
 * before later ones at the same (default) priority.
 */
export default function SingularitySphere() {
  const meshRef = useRef();
  const { gl, scene, camera, size } = useThree();
  const fbo = useFBO(size.width, size.height);

  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader,
        fragmentShader,
        uniforms: {
          uBackdrop: { value: null },
          uColorTeal: { value: new THREE.Color(PALETTE.tealBright) },
          uColorWhite: { value: new THREE.Color(PALETTE.white) },
          uTime: { value: 0 },
          uGravity: { value: 0 },
        },
      }),
    []
  );

  useFrame((_, delta) => {
    const mesh = meshRef.current;
    if (!mesh) return;

    mesh.visible = false;
    gl.setRenderTarget(fbo);
    gl.render(scene, camera);
    gl.setRenderTarget(null);
    mesh.visible = true;

    material.uniforms.uBackdrop.value = fbo.texture;
    material.uniforms.uTime.value += delta;
    material.uniforms.uGravity.value = timelineUniforms.gravity;
  });

  return (
    <mesh ref={meshRef} material={material}>
      <icosahedronGeometry args={[1.4, 6]} />
    </mesh>
  );
}
