import { defineConfig } from "tsup";

export default defineConfig({
  entry: [
    "src/index.ts",
    "src/optimization/index.ts",
    "src/evidential/index.ts",
    "src/security/index.ts",
    "src/learning/index.ts",
  ],
  format: ["cjs", "esm"],
  dts: true,
  splitting: false,
  sourcemap: true,
  clean: true,
  treeshake: true,
  minify: false,
  external: [],
  banner: {
    js: `/**
 * 🔱 ZKAEDI PRIME — Recursively Coupled Hamiltonian Framework
 * @version 1.0.0
 * @license MIT
 */`,
  },
});
