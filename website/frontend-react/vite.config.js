import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// GLSL shader files are imported with the `?raw` suffix (e.g.
// `import frag from "./x.frag.glsl?raw"`) — Vite serves any file as a
// plain string with that suffix out of the box, so no shader-specific
// plugin is required to keep the shader source in its own `.glsl` files.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
  },
});
