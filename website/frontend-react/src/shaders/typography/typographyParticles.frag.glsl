varying vec2 vUv;
varying float vSeed;
varying float vPulled;

uniform vec3 uColorTeal;
uniform vec3 uColorWhite;
uniform float uFlash;

void main() {
  vec2 c = vUv - 0.5;
  float d = length(c) * 2.0;
  float glow = smoothstep(1.0, 0.0, d);
  if (glow <= 0.001) discard;

  vec3 color = mix(uColorTeal, uColorWhite, clamp(vPulled * 1.3, 0.0, 1.0));
  color = mix(color, uColorWhite, uFlash);

  float alpha = glow * (0.55 + vSeed * 0.4);
  gl_FragColor = vec4(color, alpha);
}
