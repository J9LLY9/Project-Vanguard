import WireframeShape from "../canvas/WireframeShape";
import { useInView } from "../../hooks/useInView";
import { PALETTE } from "../../lib/constants";

// TODO(content): replace with real product copy per feature.
const FEATURES = [
  {
    title: "Automatic kernel fusion",
    description: "TODO(copy): real description of how Vanguard fuses/optimizes kernels for local inference.",
    shape: "icosahedron",
    accentColor: PALETTE.accent,
  },
  {
    title: "Memory-aware scheduling",
    description: "TODO(copy): real description of Vanguard's memory bandwidth optimization.",
    shape: "cube",
    accentColor: PALETTE.accentDim,
  },
  {
    title: "Zero-config local runtime",
    description: "TODO(copy): real description of the effortless setup story.",
    shape: "torus",
    accentColor: PALETTE.accent,
  },
];

export default function FeatureGrid() {
  const [ref, inView] = useInView();

  return (
    <section ref={ref} className={`feature-grid fade-in-section${inView ? " is-visible" : ""}`} id="features">
      {FEATURES.map((feature) => (
        <div className="feature-card" key={feature.title}>
          <WireframeShape shape={feature.shape} accentColor={feature.accentColor} className="feature-card__visual" />
          <h3 className="feature-card__title">{feature.title}</h3>
          <p className="feature-card__description">{feature.description}</p>
        </div>
      ))}
    </section>
  );
}
