// Regular GLSL ES 1.00 (plain THREE.ShaderMaterial, not Raw/GLSL3) —
// unlike the dust cloud, the tagline particles don't need `gl_InstanceID`
// because they don't read from a GPGPU simulation texture at all. Their
// base positions (sampled from canvas-rendered typography — see
// lib/typographyParticles.js) are baked once into a per-instance
// `InstancedBufferAttribute`, and the pull-toward-center is computed
// ANALYTICALLY here from `uGravity`, not integrated from a persisted
// velocity. That's a deliberate, smaller tool for a smaller job: this
// particle set only exists for the span of Act 2 and needs no
// frame-to-frame physics memory, just "where is this point right now,
// given how far into the collapse we are" — see Teacher Mode in
// components/canvas/TaglineParticles.jsx for the fuller reasoning.
attribute vec3 aBasePosition;
attribute float aSeed;

uniform float uGravity;
uniform float uBaseSize;

varying vec2 vUv;
varying float vSeed;
varying float vPulled;

void main() {
  vec3 toCenter = -aBasePosition;
  float pulled = uGravity * uGravity; // ease-in: barely moves at first, rushes in toward the end
  vec3 instancePos = aBasePosition + toCenter * pulled * 0.94;

  vec3 cameraRight = vec3(modelViewMatrix[0][0], modelViewMatrix[1][0], modelViewMatrix[2][0]);
  vec3 cameraUp = vec3(modelViewMatrix[0][1], modelViewMatrix[1][1], modelViewMatrix[2][1]);

  // Same "stretch along the direction of travel, don't just scale"
  // requirement as the dust cloud — here the direction is analytic
  // (straight toward the origin) rather than sampled from a velocity
  // texture, since these particles have no independent velocity.
  vec3 towardCenterDir = length(toCenter) > 0.0001 ? normalize(toCenter) : vec3(1.0, 0.0, 0.0);
  vec2 dirLocal = normalize(vec2(dot(towardCenterDir, cameraRight), dot(towardCenterDir, cameraUp)) + 1e-5);
  vec2 dirPerp = vec2(-dirLocal.y, dirLocal.x);

  float elongation = 1.0 + pulled * 3.5;
  float thinning = 1.0 / sqrt(elongation);

  vec2 local = position.xy * uBaseSize * (0.7 + aSeed * 0.6);
  vec2 stretched = dirLocal * local.x * elongation + dirPerp * local.y * thinning;
  vec3 worldOffset = cameraRight * stretched.x + cameraUp * stretched.y;

  vUv = uv;
  vSeed = aSeed;
  vPulled = pulled;

  gl_Position = projectionMatrix * modelViewMatrix * vec4(instancePos + worldOffset, 1.0);
}
