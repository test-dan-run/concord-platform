/// <reference types="vitest/config" />
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// `npm run dev` proxies the API and gateway to a catalog running on CATALOG_PORT (default 8700).
const target = `http://127.0.0.1:${process.env.CATALOG_PORT || 8700}`;

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 5175, proxy: { "/api": { target }, "/m": { target } } },
  build: { outDir: "dist", sourcemap: false, chunkSizeWarningLimit: 900 },
  test: { environment: "jsdom", include: ["src/**/*.test.{ts,tsx}"], globals: false },
});
