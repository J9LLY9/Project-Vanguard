import WireframeReveal from "../../layout/WireframeReveal";

/**
 * One feature block: copy on one side, a WireframeReveal shape on the
 * other (see components/layout/WireframeReveal.jsx for the reveal
 * mechanics themselves — this component is just layout + copy around it).
 * `reverse` alternates which side the visual sits on, for the usual
 * "zig-zag" feature-list layout.
 */
export default function FeatureSection({ title, description, shape, accentColor, reverse = false }) {
  return (
    <section className={`feature-section${reverse ? " feature-section--reverse" : ""}`}>
      <div className="feature-section__copy">
        <h3 className="feature-section__title">{title}</h3>
        <p className="feature-section__description">{description}</p>
      </div>
      <WireframeReveal shape={shape} accentColor={accentColor} className="feature-section__visual" />
    </section>
  );
}
