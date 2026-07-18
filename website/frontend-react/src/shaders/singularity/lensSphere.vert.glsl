varying vec3 vNormal;
varying vec3 vViewPosition;
varying vec2 vScreenUv;

void main() {
  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  vViewPosition = -mvPosition.xyz;
  vNormal = normalize(normalMatrix * normal);

  vec4 clipPos = projectionMatrix * mvPosition;
  gl_Position = clipPos;
  vScreenUv = (clipPos.xy / clipPos.w) * 0.5 + 0.5;
}
