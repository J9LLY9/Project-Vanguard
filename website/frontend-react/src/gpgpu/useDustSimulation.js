import { useEffect, useMemo } from "react";
import { useThree } from "@react-three/fiber";
import { GPUComputationRenderer } from "three/examples/jsm/misc/GPUComputationRenderer.js";
import velocityShaderSource from "../shaders/gpgpu/gpgpuVelocity.frag.glsl?raw";
import positionShaderSource from "../shaders/gpgpu/gpgpuPosition.frag.glsl?raw";
import noiseGLSL from "../shaders/common/noise.glsl?raw";
import { generateDustPositions } from "../lib/particleDistribution";

const velocityShader = `${noiseGLSL}\n${velocityShaderSource}`;

// Positions come from the SHARED seeded generator (lib/particleDistribution.js)
// rather than an inline Math.random() fill — components/canvas/DataLines.jsx
// needs a CPU-readable copy of these exact same positions to compute
// nearest-neighbor pairs, and only a deterministic generator lets both
// call sites agree on where every particle starts.
function fillInitialPositions(texture, textureSize, count) {
  texture.image.data.set(generateDustPositions(count, textureSize));
}

function fillInitialVelocities(texture, textureSize) {
  const data = texture.image.data;
  const total = textureSize * textureSize;
  for (let i = 0; i < total; i++) {
    const s = i * 4;
    data[s] = (Math.random() - 0.5) * 0.02;
    data[s + 1] = (Math.random() - 0.5) * 0.02;
    data[s + 2] = (Math.random() - 0.5) * 0.02;
    data[s + 3] = 0;
  }
}

/**
 * GPGPU ping-pong FBO simulation for the Act 1/2 dust cloud's
 * gravitational-attraction physics. Two render-target-backed textures
 * ("variables" in GPUComputationRenderer's vocabulary) — position and
 * velocity — each read last frame's OWN and the OTHER variable's state
 * and write this frame's update; GPUComputationRenderer swaps each
 * variable's read/write targets every `compute()` call, which is the
 * "ping-pong" part (frame N's output texture becomes frame N+1's input,
 * alternating between two physical render targets so you're never
 * reading and writing the same texture in one draw call).
 *
 * Deliberately returns an IMPERATIVE `step()` rather than running its
 * own `useFrame` — this simulation is shared by TWO renderers (the dust
 * points AND the data-line connections, see components/canvas/DustSystem.jsx),
 * so exactly one owner (DustSystem) must call `step()` once per frame,
 * before either renderer's own frame callback reads the resulting
 * textures — DustSystem's Teacher Mode note explains why that ordering
 * falls out of React's render order for free, with no custom `useFrame`
 * priority required. Calling `step()` from more than one place would
 * advance the simulation twice per rendered frame.
 */
export function useDustSimulation({ count, textureSize }) {
  const { gl } = useThree();

  const gpu = useMemo(() => {
    const renderer = new GPUComputationRenderer(textureSize, textureSize, gl);

    const dtPosition = renderer.createTexture();
    const dtVelocity = renderer.createTexture();
    fillInitialPositions(dtPosition, textureSize, count);
    fillInitialVelocities(dtVelocity, textureSize);

    const positionVariable = renderer.addVariable("texturePosition", positionShaderSource, dtPosition);
    const velocityVariable = renderer.addVariable("textureVelocity", velocityShader, dtVelocity);

    renderer.setVariableDependencies(positionVariable, [positionVariable, velocityVariable]);
    renderer.setVariableDependencies(velocityVariable, [positionVariable, velocityVariable]);

    velocityVariable.material.uniforms.uGravity = { value: 0 };
    velocityVariable.material.uniforms.uTime = { value: 0 };
    velocityVariable.material.uniforms.uDeltaTime = { value: 0 };
    positionVariable.material.uniforms.uDeltaTime = { value: 0 };

    const error = renderer.init();
    if (error !== null) {
      // eslint-disable-next-line no-console
      console.error("[useDustSimulation] GPUComputationRenderer failed to init:", error);
    }

    return { renderer, positionVariable, velocityVariable };
  }, [gl, count, textureSize]);

  useEffect(() => {
    return () => {
      gpu.renderer.dispose?.();
    };
  }, [gpu]);

  function step(delta, gravity) {
    // Clamp delta so a tab-switch/GC stutter doesn't fling every
    // particle outward in one giant integration step.
    const dt = Math.min(delta, 1 / 30);
    gpu.velocityVariable.material.uniforms.uGravity.value = gravity;
    gpu.velocityVariable.material.uniforms.uDeltaTime.value = dt;
    gpu.velocityVariable.material.uniforms.uTime.value += dt;
    gpu.positionVariable.material.uniforms.uDeltaTime.value = dt;
    gpu.renderer.compute();
  }

  function getPositionTexture() {
    return gpu.renderer.getCurrentRenderTarget(gpu.positionVariable).texture;
  }

  function getVelocityTexture() {
    return gpu.renderer.getCurrentRenderTarget(gpu.velocityVariable).texture;
  }

  return { step, getPositionTexture, getVelocityTexture, textureSize };
}
