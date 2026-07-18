import { useSceneStore } from "../../state/sceneStore";
import { NAV_LINKS } from "../../lib/constants";

/**
 * "Persistent nav ... that fades in only after Act 3 completes." Reads
 * `pinComplete` from the shared store — set true by
 * hooks/useScrollTimeline.js once the Acts 1-3 pin's progress reaches
 * 1 — rather than any local scroll math of its own, so this component
 * can never disagree with the Canvas about when Act 3 has actually
 * resolved.
 */
export default function Nav() {
  const pinComplete = useSceneStore((s) => s.pinComplete);

  return (
    <nav className={`nav${pinComplete ? " nav--visible" : ""}`} aria-hidden={!pinComplete}>
      <span className="nav__wordmark">Vanguard</span>
      <ul className="nav__links">
        {NAV_LINKS.map((link) => (
          <li key={link.href}>
            <a href={link.href}>{link.label}</a>
          </li>
        ))}
      </ul>
    </nav>
  );
}
