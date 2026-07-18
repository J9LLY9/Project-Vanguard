// No manual `#version 300 es` — see neuralDust.vert.glsl's header comment.
precision highp float;

in vec2 vUv;
in float vSeed;
in float vSpeed;

uniform vec3 uColorTeal;
uniform vec3 uColorObsidian;
uniform vec3 uColorWhite;
uniform float uGravity;
uniform float uFlash;

out vec4 fragColor;

void main() {
  vec2 c = vUv - 0.5;
  float d = length(c) * 2.0;
  float glow = smoothstep(1.0, 0.0, d);
  if (glow <= 0.001) discard;

  // Strict palette: obsidian <-> teal per-particle (seeded, not random
  // per-frame, so a given particle keeps its identity as it moves),
  // hot-white only as a highlight on fast-moving particles once gravity
  // has taken hold, and the Act 3 flash sweeps everything to white.
  vec3 base = mix(uColorObsidian, uColorTeal, 0.35 + vSeed * 0.65);
  vec3 hot = mix(base, uColorWhite, clamp(vSpeed * 1.5, 0.0, 1.0) * uGravity);
  vec3 color = mix(hot, uColorWhite, uFlash);

  float alpha = glow * (0.35 + vSeed * 0.5);
  fragColor = vec4(color, alpha);
}
