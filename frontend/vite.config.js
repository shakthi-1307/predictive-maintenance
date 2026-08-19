import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// @vitejs/plugin-react is declared in package.json but was never
// wired up, so the dev server had no Fast Refresh.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
  },
});
