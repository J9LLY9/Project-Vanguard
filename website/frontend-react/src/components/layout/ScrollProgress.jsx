import { useSceneStore } from "../../state/sceneStore";

/**
 * "Scroll-progress indicator ... spanning all five acts." Reads
 * `pageProgress` (see hooks/useScrollTimeline.js's page-wide, unpinned
 * ScrollTrigger) — deliberately NOT `pinnedProgress`, which only covers
 * Acts 1-3 and would freeze at 1 for the entire back half of the page.
 */
export default function ScrollProgress() {
  const pageProgress = useSceneStore((s) => s.pageProgress);
  const percent = Math.round(pageProgress * 100);

  return (
    <div className="scroll-progress" aria-hidden="true">
      <div className="scroll-progress__track">
        <div className="scroll-progress__fill" style={{ height: `${percent}%` }} />
        <div className="scroll-progress__dot" style={{ top: `${percent}%` }} />
      </div>
    </div>
  );
}
