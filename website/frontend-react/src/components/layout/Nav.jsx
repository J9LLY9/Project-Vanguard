import { NAV_LINKS } from "../../lib/constants";

// Visible immediately on load — no fade-in gate, no dependency on
// anything below it having scrolled/resolved.
export default function Nav() {
  return (
    <nav className="nav">
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
