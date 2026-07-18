// No manual `#version 300 es` — the material sets `glslVersion:
// THREE.GLSL3`, which prepends it; see neuralDust.vert.glsl for why.
//
// Faint data-line connections between nearby particles. The PAIRS
// (which particle connects to which) are computed once, on the CPU, at
// mount time — see components/canvas/DataLines.jsx — because a true
// per-frame nearest-neighbor search across 10-20k particles is an O(n²)
// job with no cheap GPU primitive for it in plain WebGL. What CAN stay
// fully GPU-driven is which particles those fixed pairs currently point
// at: each line vertex carries only an integer index (`aParticleIndex`)
// into the SAME position texture the dust cloud reads, so as the
// simulation moves particles around, every line's endpoints move with
// them for free, with zero per-frame CPU work — only the topology
// (which N pairs exist) is fixed at spawn, not the endpoints' positions.
precision highp float;

in float aParticleIndex;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform sampler2D uPositionTexture;
uniform float uTextureSize;

void main() {
  float col = mod(aParticleIndex, uTextureSize);
  float row = floor(aParticleIndex / uTextureSize);
  vec2 uv = (vec2(col, row) + 0.5) / uTextureSize;
  vec3 particlePos = texture(uPositionTexture, uv).xyz;

  gl_Position = projectionMatrix * modelViewMatrix * vec4(particlePos, 1.0);
}
