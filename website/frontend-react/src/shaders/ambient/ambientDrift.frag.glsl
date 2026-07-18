uniform vec3 uColor;

void main() {
  vec2 c = gl_PointCoord - 0.5;
  float d = length(c) * 2.0;
  float glow = smoothstep(1.0, 0.0, d);
  if (glow <= 0.001) discard;
  gl_FragColor = vec4(uColor, glow * 0.4);
}
