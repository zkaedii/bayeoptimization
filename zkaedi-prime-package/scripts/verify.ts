#!/usr/bin/env node
/**
 * 🔱 ZKAEDI PRIME — Package Verification Script
 * 
 * Verifies package is ready for publishing
 */

import { readFileSync, existsSync } from "fs";
import { join } from "path";

const errors: string[] = [];
const warnings: string[] = [];

function checkFile(path: string, required: boolean = true): void {
  if (!existsSync(path)) {
    if (required) {
      errors.push(`❌ Missing required file: ${path}`);
    } else {
      warnings.push(`⚠️  Missing optional file: ${path}`);
    }
  } else {
    console.log(`✅ Found: ${path}`);
  }
}

function checkPackageJson(): void {
  const pkgPath = join(process.cwd(), "package.json");
  if (!existsSync(pkgPath)) {
    errors.push("❌ package.json not found");
    return;
  }

  const pkg = JSON.parse(readFileSync(pkgPath, "utf-8"));

  // Check required fields
  if (!pkg.name) errors.push("❌ package.json missing 'name'");
  if (!pkg.version) errors.push("❌ package.json missing 'version'");
  if (!pkg.main) errors.push("❌ package.json missing 'main'");
  if (!pkg.types) errors.push("❌ package.json missing 'types'");
  if (!pkg.exports) errors.push("❌ package.json missing 'exports'");

  // Check name format
  if (!pkg.name.startsWith("@zkaedi/")) {
    warnings.push("⚠️  Package name should start with @zkaedi/");
  }

  console.log(`✅ package.json is valid`);
  console.log(`   Name: ${pkg.name}`);
  console.log(`   Version: ${pkg.version}`);
}

function main(): void {
  console.log("🔱 ZKAEDI PRIME Package Verification\n");
  console.log("=".repeat(50));

  // Check required files
  console.log("\n📦 Checking required files...");
  checkFile("package.json");
  checkFile("tsconfig.json");
  checkFile("tsup.config.ts");
  checkFile("README.md");
  checkFile("LICENSE");
  checkFile("CHANGELOG.md");
  checkFile(".npmignore");
  checkFile(".gitignore");

  // Check source files
  console.log("\n📝 Checking source files...");
  checkFile("src/index.ts");
  checkFile("src/optimization/index.ts");
  checkFile("src/evidential/index.ts");
  checkFile("src/security/index.ts");
  checkFile("src/learning/index.ts");

  // Check package.json
  console.log("\n📋 Checking package.json...");
  checkPackageJson();

  // Summary
  console.log("\n" + "=".repeat(50));
  console.log("\n📊 Summary:\n");

  if (errors.length === 0 && warnings.length === 0) {
    console.log("✅ All checks passed! Package is ready.\n");
    process.exit(0);
  }

  if (warnings.length > 0) {
    console.log("⚠️  Warnings:");
    warnings.forEach((w) => console.log(`   ${w}`));
    console.log();
  }

  if (errors.length > 0) {
    console.log("❌ Errors:");
    errors.forEach((e) => console.log(`   ${e}`));
    console.log();
    process.exit(1);
  }

  process.exit(0);
}

main();
