import { Canvas } from "@react-three/fiber";
import AmbientParticles from "../canvas/AmbientParticles";
import OptimizationProof from "./act4/OptimizationProof";
import FeatureSection from "./act4/FeatureSection";
import { PALETTE } from "../../lib/constants";

// ============================================================================
// ACT 4 — "THE NEW WORLD." Flagged per the brief as needing the most
// build-out of the five acts. Everything below this line that isn't the
// AmbientParticles background canvas or the wireframe-to-solid
// FeatureSection mechanic itself is PLACEHOLDER content: copy, feature
// list, icons/shapes. The structural pieces (scroll behavior, the
// recurring reveal transition, the ambient motif, the optimization-proof
// component's animation) are real and functional; what's missing is
// real product content to pour into them.
//
// TODO(content): replace FEATURES below with real product copy + a
// deliberate choice of shape/accentColor per feature (right now they
// just cycle through the strict palette + a few primitive shapes so the
// section isn't empty).
// TODO(design): FeatureSection's zig-zag layout is a reasonable default,
// not a locked-in decision — revisit once there's real content to see
// if every feature actually wants the same treatment.
// TODO(data): OptimizationProof.jsx's stats are placeholders — see that
// file's own TODO for what's needed to replace them.
// TODO(assets): OptimizationProof could use a visual diagram alongside
// the numeric bars; nothing here blocks adding one once it exists.
// ============================================================================

const FEATURES = [
  {
    title: "Automatic kernel fusion",
    description:
      "TODO(copy): real description of how Vanguard fuses/optimizes kernels for local inference.",
    shape: "icosahedron",
    accentColor: PALETTE.teal,
  },
  {
    title: "Memory-aware scheduling",
    description: "TODO(copy): real description of Vanguard's memory bandwidth optimization.",
    shape: "box",
    accentColor: PALETTE.tealBright,
    reverse: true,
  },
  {
    title: "Zero-config local runtime",
    description: "TODO(copy): real description of the effortless setup story.",
    shape: "torus",
    accentColor: PALETTE.teal,
  },
];

export default function Act4NewWorld() {
  return (
    <div className="act4">
      {/* Low-density, no-physics ambient motif — see AmbientParticles.jsx.
          Fixed behind the natural-scroll content for the rest of the page. */}
      <Canvas className="act4__ambient-canvas" dpr={[1, 1.5]} camera={{ position: [0, 0, 12], fov: 45 }}>
        <AmbientParticles />
      </Canvas>

      <OptimizationProof />

      <div className="act4__features">
        {FEATURES.map((feature) => (
          <FeatureSection key={feature.title} {...feature} />
        ))}
      </div>
    </div>
  );
}
