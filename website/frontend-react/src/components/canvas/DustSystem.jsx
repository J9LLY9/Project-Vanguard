import { useFrame } from "@react-three/fiber";
import { useDustSimulation } from "../../gpgpu/useDustSimulation";
import { timelineUniforms } from "../../hooks/useScrollTimeline";
import NeuralDust from "./NeuralDust";
import DataLines from "./DataLines";

/**
 * Owns the ONE GPGPU simulation shared by both the dust points and the
 * data-line connections, and is the ONLY place that calls `gpu.step()` —
 * see gpgpu/useDustSimulation.js's Teacher Mode note.
 *
 * Deliberately uses r3f's DEFAULT `useFrame` priority (no explicit
 * priority argument), not a custom one: passing ANY non-zero priority
 * ANYWHERE in the Canvas tree switches off react-three-fiber's automatic
 * per-frame render — at that point SOMETHING has to call
 * `gl.render(scene, camera)` manually, every frame, or the canvas goes
 * blank. That trade is worth it for SingularitySphere's render-to-texture
 * pass (see that file), but not needed here: React already guarantees a
 * parent component's `useFrame` registers (and therefore, at equal
 * priority, RUNS) before its children's — `DustSystem` renders before
 * `NeuralDust`/`DataLines` mount, full stop — so ordering "step before
 * read" falls out of the component tree shape for free.
 */
export default function DustSystem({ count, textureSize }) {
  const gpu = useDustSimulation({ count, textureSize });

  useFrame((_, delta) => {
    gpu.step(delta, timelineUniforms.gravity);
  });

  return (
    <group>
      <NeuralDust gpu={gpu} count={count} textureSize={textureSize} />
      <DataLines gpu={gpu} count={count} textureSize={textureSize} />
    </group>
  );
}
