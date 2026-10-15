import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run build` writes the page to ../site, which backend.py serves at /.
// `npm run dev` serves it on 5173 and forwards /api to backend.py on 8000.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../site",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
});
