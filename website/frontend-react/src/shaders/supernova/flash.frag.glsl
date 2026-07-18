// Act 3's white burst — entirely shader-driven (radial burst shape +
// rotating ray mask), not a video asset. `uFlash` is a single 0..1
// progress value for the whole act; the burst/clear curve below is
// deliberately shaped as ONE continuous function of it rather than two
// separately-triggered animations, so scrubbing (or a slow device
// skipping frames) can never land between "burst" and "clear" in an
// inconsistent state — see hooks/useScrollTimeline.js for how uFlash
// is driven.
varying vec2 vUv;

uniform float uFlash;
uniform float uTime;
uniform vec3 uColorTeal;
uniform vec3 uColorWhite;

void main() {
  vec2 c = vUv - 0.5;
  float d = length(c) * 2.0;

  float angle = atan(c.y, c.x);
  float rays = 0.5 + 0.5 * sin(angle * 10.0 + uTime * 0.6);
  float rayMask = pow(rays, 3.0) * smoothstep(1.3, 0.0, d);

  float core = smoothstep(1.3, 0.0, d);
  float burstShape = clamp(core + rayMask * 0.6, 0.0, 1.0);

  // Rises as the burst begins, peaks near-solid white, then clears back
  // to fully transparent — "bursts ... then clears to reveal" as one
  // triangular envelope over uFlash.
  float buildUp = smoothstep(0.0, 0.35, uFlash);
  float clear = 1.0 - smoothstep(0.42, 1.0, uFlash);
  float intensity = min(buildUp, clear);

  // Near the peak, wash the whole frame to flat white rather than just
  // brightening the ray pattern — that's what reads as a genuine
  // blackout/whiteout instant instead of "the rays got brighter."
  float wash = smoothstep(0.7, 1.0, intensity);
  float alpha = mix(burstShape * intensity, 1.0, wash);

  vec3 color = mix(uColorTeal, uColorWhite, clamp(intensity * 1.6, 0.0, 1.0));

  gl_FragColor = vec4(color, clamp(alpha, 0.0, 1.0));
}
