import type { Config } from "jest";

const config: Config = {
  preset: "ts-jest",
  testEnvironment: "node",
  roots: ["<rootDir>/tests"],
  testMatch: ["**/test_server.ts"],
  moduleFileExtensions: ["ts", "js", "json"],
  transform: {
    "^.+\\.ts$": ["ts-jest", {
      tsconfig: {
        esModuleInterop: true,
        strict: true,
        module: "commonjs",
        target: "ES2022",
        moduleResolution: "node",
      },
    }],
  },
  coverageDirectory: "coverage",
  collectCoverageFrom: [
    "src/server/**/*.ts",
    "!src/server/tsconfig.json",
  ],
  coverageThreshold: {
    global: {
      lines: 95,
    },
  },
};

export default config;
