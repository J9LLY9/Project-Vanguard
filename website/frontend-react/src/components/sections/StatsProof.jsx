import { useEffect, useRef } from "react";
import { useInView } from "../../hooks/useInView";
import { smoothstep } from "../../lib/mathUtils";

// TODO(real-data): every number below is a placeholder. Replace with
// real benchmark output (and cite the benchmark methodology somewhere
// near this section — "reduced by 61%" needs a footnote on WHAT was
// measured, on WHAT hardware, before it's a credible marketing claim).
const DEFAULT_STATS = [
  { label: "Memory bandwidth", before: 100, after: 39, unit: "% baseline", lowerIsBetter: true },
  { label: "Kernel execution time", before: 100, after: 44, unit: "% baseline", lowerIsBetter: true },
  { label: "Tokens / second", before: 18, after: 61, unit: "tok/s", lowerIsBetter: false },
];

const FILL_DURATION_MS = 900;

function StatBar({ stat, inView }) {
  const fillRef = useRef();
  const valueRef = useRef();
  const played = useRef(false);

  useEffect(() => {
    if (!inView || played.current) return;
    played.current = true;

    const changeRatio = stat.after / stat.before;
    const fillScale = stat.lowerIsBetter ? changeRatio : 1;
    const start = performance.now();

    // Fire-once tween on its own clock — plays out fully regardless of
    // further scrolling, and never reverses.
    function tick(now) {
      const t = smoothstep(Math.min(1, (now - start) / FILL_DURATION_MS));
      if (fillRef.current) fillRef.current.style.transform = `scaleY(${fillScale * t})`;
      if (valueRef.current) {
        valueRef.current.textContent = Math.round(stat.before + (stat.after - stat.before) * t);
      }
      if (t < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }, [inView, stat]);

  return (
    <div className="stat-bar">
      <div className="stat-bar__track">
        <div className="stat-bar__baseline" />
        <div ref={fillRef} className="stat-bar__fill" style={{ transform: "scaleY(0)" }} />
      </div>
      <p className="stat-bar__value">
        <span ref={valueRef}>{stat.before}</span>
        <span className="stat-bar__unit">{stat.unit}</span>
      </p>
      <p className="stat-bar__label">{stat.label}</p>
    </div>
  );
}

/**
 * "Proof, not promises." — animated stat bars that fill 0 to value once
 * when the section enters the viewport. No scrubbing, no reverse on
 * scroll-up; each bar's fill is a fixed-duration tween fired a single
 * time via useInView.
 */
export default function StatsProof({ stats = DEFAULT_STATS }) {
  const [ref, inView] = useInView();

  return (
    <section ref={ref} className={`optimization-proof fade-in-section${inView ? " is-visible" : ""}`} id="benchmarks">
      <h2 className="section-heading">Proof, not promises.</h2>
      <p className="section-subheading">
        {/* TODO(copy): replace with real methodology summary + link to full benchmark writeup. */}
        Measured against baseline inference on representative local-model workloads.
      </p>
      <div className="optimization-proof__stats">
        {stats.map((stat) => (
          <StatBar key={stat.label} stat={stat} inView={inView} />
        ))}
      </div>
    </section>
  );
}
