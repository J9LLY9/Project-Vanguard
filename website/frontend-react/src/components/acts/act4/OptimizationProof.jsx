import { useEffect, useRef } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

gsap.registerPlugin(ScrollTrigger);

// TODO(real-data): every number below is a placeholder. Replace with
// real benchmark output (and cite the benchmark methodology somewhere
// near this section — "reduced by 61%" needs a footnote on WHAT was
// measured, on WHAT hardware, before it's a credible marketing claim).
const DEFAULT_STATS = [
  { label: "Memory bandwidth", before: 100, after: 39, unit: "% baseline", lowerIsBetter: true },
  { label: "Kernel execution time", before: 100, after: 44, unit: "% baseline", lowerIsBetter: true },
  { label: "Tokens / second", before: 18, after: 61, unit: "tok/s", lowerIsBetter: false },
];

function StatBar({ stat }) {
  const fillRef = useRef();
  const wrapperRef = useRef();
  const valueRef = useRef();

  useEffect(() => {
    if (!wrapperRef.current) return undefined;

    const changeRatio = stat.after / stat.before;
    const fillScale = stat.lowerIsBetter ? changeRatio : 1; // "after" bar height relative to "before"'s baseline of 1

    const trigger = ScrollTrigger.create({
      trigger: wrapperRef.current,
      start: "top 75%",
      once: true,
      onEnter: () => {
        const counter = { value: stat.before };
        gsap.timeline()
          .to(fillRef.current, { scaleY: fillScale, duration: 1.1, ease: "power3.out" }, 0)
          .to(counter, {
            value: stat.after,
            duration: 1.1,
            ease: "power3.out",
            onUpdate: () => {
              if (valueRef.current) valueRef.current.textContent = Math.round(counter.value);
            },
          }, 0);
      },
    });

    return () => trigger.kill();
  }, [stat]);

  return (
    <div ref={wrapperRef} className="stat-bar">
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
 * "Optimization proof section: animated before/after showing memory
 * bandwidth reduction and kernel speedup." Accepts `stats` as a prop
 * specifically so the placeholder numbers above are trivial to swap out
 * without touching the animation logic — see DEFAULT_STATS' TODO.
 *
 * TODO(content): this only covers the numeric proof. The brief also
 * implies a visual "before/after" — consider pairing this with a
 * WireframeReveal-driven diagram (see components/layout/WireframeReveal.jsx)
 * showing, e.g., a kernel call graph shrinking, once there's a real
 * asset/diagram to drive it from.
 */
export default function OptimizationProof({ stats = DEFAULT_STATS }) {
  return (
    <section className="optimization-proof" id="benchmarks">
      <h2 className="section-heading">Proof, not promises.</h2>
      <p className="section-subheading">
        {/* TODO(copy): replace with real methodology summary + link to full benchmark writeup. */}
        Measured against baseline inference on representative local-model workloads.
      </p>
      <div className="optimization-proof__stats">
        {stats.map((stat) => (
          <StatBar key={stat.label} stat={stat} />
        ))}
      </div>
    </section>
  );
}
