// TODO(content): CTA copy, real download/signup destinations, footer
// link structure — all placeholder below. Wire up the actual signup
// flow (email capture? OAuth? direct binary download per-platform?)
// once that's decided; the button below is inert.
export default function Act5Invitation() {
  return (
    <section className="invitation" id="get-started">
      <h2 className="invitation__title">Effortless starts here.</h2>
      <p className="invitation__subtitle">
        {/* TODO(copy): real supporting line for the final CTA. */}
        Download Vanguard and run your first optimized local model in minutes.
      </p>
      <div className="invitation__actions">
        {/* TODO(cta): wire to the real download/signup destination. */}
        <a className="button button--primary" href="#">
          Download Vanguard
        </a>
        <a className="button button--ghost" href="#docs">
          Read the docs
        </a>
      </div>

      <footer className="footer">
        <span className="footer__wordmark">Vanguard</span>
        <ul className="footer__links">
          {/* TODO(links): real footer nav — legal, social, docs, changelog. */}
          <li>
            <a href="#product">Product</a>
          </li>
          <li>
            <a href="#benchmarks">Benchmarks</a>
          </li>
          <li>
            <a href="#docs">Docs</a>
          </li>
        </ul>
        <span className="footer__copyright">© {new Date().getFullYear()} Vanguard</span>
      </footer>
    </section>
  );
}
