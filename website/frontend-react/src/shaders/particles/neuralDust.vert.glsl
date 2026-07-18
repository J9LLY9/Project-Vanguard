// GLSL ES 3.00 (WebGL2) required for `gl_InstanceID` — see the comment
// in components/canvas/NeuralDust.jsx for why this is a RawShaderMaterial
// rather than a plain ShaderMaterial. NOTE: no manual `#version 300 es`
// here — the material is created with `glslVersion: THREE.GLSL3`, which
// makes Three.js prepend that line itself; adding it again here would
// produce two `#version` directives and fail to compile.
//
// This renders each of the `count` dust particles as a camera-facing
// quad (2 triangles) whose position, size, and — critically — ANISOTROPIC
// STRETCH come entirely from sampling the GPGPU position/velocity
// textures via `gl_InstanceID`. There is no `instanceMatrix` use at all:
// the InstancedMesh's per-instance transforms are left at their default
// identity and never touched from JS, because every particle's true
// transform is derived here, on the GPU, from the simulation textures —
// which is what makes this "no per-frame JS position loops" rather than
// just "no per-frame JS loop that happens to also update a scalar."
precision highp float;

in vec3 position; // base quad corner, local -0.5..0.5 on XY
in vec2 uv;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;

uniform sampler2D uPositionTexture;
uniform sampler2D uVelocityTexture;
uniform float uTextureSize;
uniform float uBaseSize;
uniform float uStretchFactor;
uniform float uMaxStretch;
uniform float uGravity;

out vec2 vUv;
out float vSeed;
out float vSpeed;

void main() {
  float col = mod(float(gl_InstanceID), uTextureSize);
  float row = floor(float(gl_InstanceID) / uTextureSize);
  vec2 instanceUV = (vec2(col, row) + 0.5) / uTextureSize;

  vec4 posData = texture(uPositionTexture, instanceUV);
  vec4 velData = texture(uVelocityTexture, instanceUV);
  vec3 particlePos = posData.xyz;
  vec3 vel = velData.xyz;
  float seed = posData.w;

  // Camera-facing basis, read straight out of the view matrix — the
  // standard cheap billboard technique (no per-instance quaternion math).
  vec3 cameraRight = vec3(modelViewMatrix[0][0], modelViewMatrix[1][0], modelViewMatrix[2][0]);
  vec3 cameraUp = vec3(modelViewMatrix[0][1], modelViewMatrix[1][1], modelViewMatrix[2][1]);

  // Project velocity onto the billboard's own plane, THEN stretch along
  // that projected direction — not a uniform scale. `thinning` keeps the
  // quad's rendered area roughly constant as it elongates, so fast
  // particles read as "stretched streaks," not "particles that got bigger."
  vec2 velLocal = vec2(dot(vel, cameraRight), dot(vel, cameraUp));
  float speed = length(velLocal);
  vec2 stretchDir = speed > 0.0001 ? velLocal / speed : vec2(1.0, 0.0);
  vec2 stretchPerp = vec2(-stretchDir.y, stretchDir.x);

  float elongation = 1.0 + clamp(speed * uStretchFactor, 0.0, uMaxStretch) * uGravity;
  float thinning = 1.0 / sqrt(elongation);

  vec2 local = position.xy * uBaseSize * (0.6 + seed * 0.8);
  vec2 stretched = stretchDir * local.x * elongation + stretchPerp * local.y * thinning;

  vec3 worldOffset = cameraRight * stretched.x + cameraUp * stretched.y;
  vec3 finalPos = particlePos + worldOffset;

  vUv = uv;
  vSeed = seed;
  vSpeed = speed;

  gl_Position = projectionMatrix * modelViewMatrix * vec4(finalPos, 1.0);
}
