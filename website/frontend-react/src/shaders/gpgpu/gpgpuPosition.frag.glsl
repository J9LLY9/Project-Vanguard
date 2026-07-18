// GPUComputationRenderer variable: "texturePosition". Simple Euler
// integration of the velocity textureVelocity just computed this same
// pass — deliberately the "dumb" half of the ping-pong pair; all the
// actual physics decisions live in gpgpuVelocity.frag.glsl so there's
// one place to tune the gravitational behavior.
uniform float uDeltaTime;

void main() {
  vec2 uv = gl_FragCoord.xy / resolution.xy;

  vec4 posData = texture2D(texturePosition, uv);
  vec4 velData = texture2D(textureVelocity, uv);

  vec3 pos = posData.xyz + velData.xyz * uDeltaTime;
  float seed = posData.w;

  gl_FragColor = vec4(pos, seed);
}
