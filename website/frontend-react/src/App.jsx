import Nav from "./components/layout/Nav";
import Hero from "./components/sections/Hero";
import StatsProof from "./components/sections/StatsProof";
import FeatureGrid from "./components/sections/FeatureGrid";
import CTAFooter from "./components/sections/CTAFooter";

/**
 * Standard scrolling page — no pinning, no scroll-jacking, no shared
 * Canvas/timeline. Each section owns its own entrance animation (see
 * hooks/useInView.js) and native scroll handles everything else.
 */
export default function App() {
  return (
    <>
      <Nav />
      <main>
        <Hero />
        <div className="section-divider" />
        <StatsProof />
        <div className="section-divider" />
        <FeatureGrid />
        <div className="section-divider" />
        <CTAFooter />
      </main>
    </>
  );
}
