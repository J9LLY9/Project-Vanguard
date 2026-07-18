// Act 4's "ambient particle motif ... low density/no physics, calm
// drift only" — a cheap per-vertex sine drift, computed here (still
// zero per-frame JS work) rather than via a GPGPU simulation. Act 4 is
// meant to feel settled after Acts 1-3's collapse; reusing the dust
// cloud's heavier machinery here would cost more than this section
// needs and would visually contradict "calm."
attribute float aSeed;
uniform float uTime;

void main() {
  vec3 p = position;
  p.x += sin(uTime * 0.15 + aSeed * 62.0) * 0.6;
  p.y += cos(uTime * 0.12 + aSeed * 40.0) * 0.6;

  vec4 mvPosition = modelViewMatrix * vec4(p, 1.0);
  gl_Position = projectionMatrix * mvPosition;
  gl_PointSize = (2.0 + aSeed * 2.0) * (60.0 / -mvPosition.z);
}
