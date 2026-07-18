/**
 * Mobile's "separate lighter-weight path" for Acts 1-3 — a looped video
 * in place of the GPGPU particle pipeline entirely, not the desktop
 * pipeline at a reduced particle count. Two reasons a video wins over
 * "just fewer particles" here: (1) `RawShaderMaterial` + `gl_InstanceID`
 * (see components/canvas/NeuralDust.jsx) needs a real WebGL2 context,
 * which is inconsistent enough across mobile browsers/GPUs to be a risk
 * for a hero moment; (2) even a WORKING reduced-count sim still means
 * running GPUComputationRenderer's ping-pong compute passes every
 * frame — non-trivial GPU work on hardware that's also thermally
 * constrained in a way desktops generally aren't. A pre-rendered video
 * sidesteps both.
 *
 * TODO(video-asset): `VIDEO_SRC` is a placeholder — swap in the real
 * export once Acts 1-3 have a final look locked on desktop (the video
 * should BE that sequence, rendered out, not a separately-art-directed
 * asset). TODO(captions): consider syncing simple DOM caption fades to
 * video currentTime via a `timeupdate` listener, mirroring
 * lib/actProgress.js's fade windows, if the mobile experience needs the
 * same act-by-act text beats the desktop DOM overlays show.
 */
const VIDEO_SRC = "/media/vanguard-transformation-placeholder.mp4";

export default function MobileHeroFallback() {
  return (
    <div className="mobile-hero">
      <video
        className="mobile-hero__video"
        src={VIDEO_SRC}
        autoPlay
        muted
        loop
        playsInline
        aria-hidden="true"
      />
      <div className="mobile-hero__overlay">
        <p className="act-caption__eyebrow">Vanguard</p>
        <h1 className="act-caption__title">Make Local AI Effortless.</h1>
      </div>
    </div>
  );
}
