// GPUComputationRenderer variable: "textureVelocity". Reads the previous
// frame's position+velocity textures (auto-bound by GPUComputationRenderer
// as `texturePosition`/`textureVelocity` samplers, named after the
// variables declared in gpgpu/useDustSimulation.js), writes this frame's
// velocity. `resolution` is auto-provided by GPUComputationRenderer.
//
// This is Act 2's "gravitational constant": `uGravity` is driven by
// ScrollTrigger (0 during Act 1, ramping 0→1 across Act 2) from plain JS
// — see hooks/useScrollTimeline.js — and read here as a uniform, so the
// PHYSICS (not just a visual scale/opacity trick) genuinely reacts to
// scroll position. Everything below runs once PER PARTICLE PER FRAME on
// the GPU; there is no per-particle JS loop anywhere in this pipeline.
uniform float uGravity;
uniform float uTime;
uniform float uDeltaTime;

void main() {
  vec2 uv = gl_FragCoord.xy / resolution.xy;

  vec4 posData = texture2D(texturePosition, uv);
  vec4 velData = texture2D(textureVelocity, uv);

  vec3 pos = posData.xyz;
  vec3 vel = velData.xyz;
  float seed = posData.w;

  // --- Gravitational pull toward the singularity at the origin ---
  vec3 toCenter = -pos;
  float dist = length(toCenter);
  vec3 pullDir = dist > 0.001 ? toCenter / dist : vec3(0.0);
  // Pull strength grows mildly with distance (capped) rather than a
  // true inverse-square law — inverse-square produces chaotic slingshot
  // orbits near the center, which reads as "particles flying past," not
  // the smooth convergent collapse the brief describes. This is a
  // deliberately "designed," not physically literal, gravity curve.
  float pullStrength = uGravity * clamp(dist * 0.16, 0.0, 3.2);
  vec3 pull = pullDir * pullStrength;

  // --- Ambient drift (Act 1's "subtle drift motion") — fades out as
  // uGravity takes over, so particles smoothly hand off from floating
  // to falling rather than the two forces fighting each other. ---
  vec3 drift = curlNoise(pos * 0.05, uTime * 0.03) * 0.06 * (1.0 - uGravity);

  vel += (pull + drift) * uDeltaTime;

  // Damping — bleeds off energy so the system settles instead of
  // building unbounded velocity, and gives the collapse a "falling
  // through something viscous" character rather than a hard vacuum pull.
  vel *= 0.985;

  gl_FragColor = vec4(vel, seed);
}
