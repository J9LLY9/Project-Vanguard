import VanguardNodeLogo from "../canvas/VanguardNodeLogo";

export default function Hero() {
  return (
    <section className="hero" id="product">
      <div className="hero__copy">
        <p className="hero__label">VANGUARD</p>
        <h1 className="hero__title">Make Local AI Effortless.</h1>
        <p className="hero__subtext">
          Vanguard optimizes and runs local models on your own hardware — no cloud, no
          config, no wrestling with kernels.
        </p>
        <a className="button button--primary" href="#get-started">
          Get Started
        </a>
        <p className="hero__tagline">Runs on your hardware.</p>
      </div>
      <div className="hero__visual">
        <VanguardNodeLogo />
      </div>
    </section>
  );
}
