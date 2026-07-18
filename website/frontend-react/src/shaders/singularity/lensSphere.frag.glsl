// The central "gravitational-lens sphere." `uBackdrop` is the scene
// rendered ONE frame with this sphere hidden — see
// components/canvas/SingularitySphere.jsx for the render-to-texture
// pass — sampled here with a UV offset that bends INWARD toward screen
// center, strongest at the sphere's silhouette (fresnel-weighted) and
// growing with `uGravity`. That bend is what makes it read as a lens
// that visibly warps what's behind it, rather than a tinted glass ball.
//
// Uses dFdx/dFdy (screen-space derivatives) for the surface-distortion
// term below — core GLSL ES 3.00 built-ins under the WebGL2 context
// Three.js r150+ uses by default, so no `GL_OES_standard_derivatives`
// extension pragma is needed (and deliberately omitted: THREE.ShaderMaterial
// prepends its own precision/define boilerplate before this source, which
// would push an `#extension` line here past the "before any non-preprocessor
// token" position the GLSL spec requires for it).
precision highp float;

uniform sampler2D uBackdrop;
uniform vec3 uColorTeal;
uniform vec3 uColorWhite;
uniform float uTime;
uniform float uGravity;

varying vec3 vNormal;
varying vec3 vViewPosition;
varying vec2 vScreenUv;

// snoise() is concatenated onto this source in JS — see
// components/canvas/SingularitySphere.jsx.

void main() {
  vec3 viewDir = normalize(vViewPosition);
  vec3 normal = normalize(vNormal);
  float fresnel = pow(1.0 - max(dot(viewDir, normal), 0.0), 2.5);

  // Slow-drifting surface distortion — screen-space derivatives of a
  // noise field give a cheap, seamless "shimmering glass" wobble without
  // needing a second noise octave or a normal map.
  float n = snoise(normal * 3.0 + vec3(0.0, 0.0, uTime * 0.08));
  vec2 distortedNormalOffset = vec2(dFdx(n), dFdy(n)) * 0.6;

  vec2 towardCenter = normalize(vec2(0.5) - vScreenUv + 1e-5);
  float bendStrength = (0.05 + fresnel * 0.12) * (1.0 + uGravity * 2.5);
  vec2 sampleUv = vScreenUv + towardCenter * bendStrength + distortedNormalOffset * 0.02;

  vec3 backdrop = texture2D(uBackdrop, clamp(sampleUv, 0.001, 0.999)).rgb;

  vec3 rim = mix(uColorTeal, uColorWhite, fresnel);
  vec3 color = mix(backdrop * 0.85, rim, fresnel * 0.6);

  gl_FragColor = vec4(color, 1.0);
}
