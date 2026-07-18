import { useSceneStore } from "../../state/sceneStore";
import { ACT_BOUNDS } from "../../lib/constants";
import { getActOpacity } from "../../lib/actProgress";

/**
 * Act 1's DOM overlay — just the wordmark/entry caption. The heavy
 * lifting (the lens sphere, the dust cloud) is all Canvas content, see
 * components/canvas/Experience.jsx; this is deliberately minimal.
 */
export default function Act1Singularity() {
  const pinnedProgress = useSceneStore((s) => s.pinnedProgress);
  const opacity = getActOpacity(pinnedProgress, ACT_BOUNDS.singularity);

  return (
    <div className="act-caption act-caption--singularity" style={{ opacity }}>
      <p className="act-caption__eyebrow">Vanguard</p>
      <h1 className="act-caption__title">Make Local AI Effortless.</h1>
    </div>
  );
}
