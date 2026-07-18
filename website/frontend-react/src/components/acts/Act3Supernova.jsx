import { useSceneStore } from "../../state/sceneStore";

/**
 * Act 3's DOM overlay — appears only once the flash has cleared past its
 * midpoint (mirrors the hard-cut visibility flip on the 3D logo itself,
 * see components/canvas/VanguardNodeLogo.jsx: "lands instantly and
 * still," so this label pops in with it rather than easing in on its
 * own separate schedule).
 */
export default function Act3Supernova() {
  const pinnedProgress = useSceneStore((s) => s.pinnedProgress);
  const visible = pinnedProgress > 0.86; // matches ACT_BOUNDS.supernova's flash midpoint region

  return (
    <div className={`act-caption act-caption--supernova${visible ? " act-caption--visible" : ""}`}>
      <p className="act-caption__eyebrow act-caption__eyebrow--dark">Vanguard Node — Online</p>
    </div>
  );
}
