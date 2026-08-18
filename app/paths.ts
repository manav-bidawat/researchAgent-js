/**
 * Repository-root resolution for the TypeScript layer.
 *
 * In:  this module's own location (works from app/ under tsx and from dist/app/ once built).
 * Out: REPO_ROOT, the directory holding package.json, bridge/, src/ and config.yaml.
 */

import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

function findRoot(start: string): string {
  let dir = start;
  while (!existsSync(join(dir, "package.json")) || !existsSync(join(dir, "bridge"))) {
    const parent = dirname(dir);
    if (parent === dir) throw new Error(`repository root not found above ${start}`);
    dir = parent;
  }
  return dir;
}

export const REPO_ROOT: string = findRoot(dirname(fileURLToPath(import.meta.url)));
