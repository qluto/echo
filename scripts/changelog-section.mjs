#!/usr/bin/env node
// Print CHANGELOG.md's section for a version (the release notes), or fail if
// there is none, so a release can't go out without its changelog.
//   node scripts/changelog-section.mjs 0.16.0 [CHANGELOG.md]
import { readFileSync } from "node:fs";

const [version, file = "CHANGELOG.md"] = process.argv.slice(2);
if (!version) {
  console.error("usage: changelog-section.mjs <version> [CHANGELOG.md]");
  process.exit(2);
}
const v = version.replace(/^v/, "");
const sections = readFileSync(file, "utf8").split(/^## /m).slice(1);
const section = sections.find((s) => s.match(/^v?(\d+\.\d+\.\d+)/)?.[1] === v);
if (!section) {
  console.error(`${file} has no "## v${v}" section. Rename "## Unreleased" to "## v${v} — <date>" before tagging.`);
  process.exit(1);
}
const body = section.split("\n").slice(1).join("\n").trim();
if (!body) {
  console.error(`The "## v${v}" section of ${file} is empty.`);
  process.exit(1);
}
console.log(body);
