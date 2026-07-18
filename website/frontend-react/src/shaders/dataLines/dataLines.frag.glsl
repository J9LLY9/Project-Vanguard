// No manual `#version 300 es` — see dataLines.vert.glsl's header comment.
precision highp float;

uniform vec3 uColor;
uniform float uOpacity;
uniform float uFlash;

out vec4 fragColor;

void main() {
  vec3 color = mix(uColor, vec3(1.0), uFlash);
  fragColor = vec4(color, uOpacity);
}
