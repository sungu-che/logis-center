// proxy 원본(801b0ce0)에 덧대는 멤버 범위 수정 — Rust 판(workers/proxy/src/search.rs)의 같은 id 와 1:1. 긴 JS 블록은 백틱 · ${} 를 거르지 않고
// 읽게 proxy_patches.txt 에 두고 여기서 { id, worker, find, replace, count } 로 바꾼다 (commerce.mjs 와 같은 꼴: "@@ id" · "<<< find" · "=== replace" · ">>>").
import fs from "node:fs";

const PATCHES = new URL("./proxy_patches.txt", import.meta.url);

function readPatches() {
  const lines = fs.readFileSync(PATCHES, "utf8").replace(/\r\n/g, "\n").split("\n");
  const out = [];
  let head = null;
  let mode = null;
  let find = [];
  let replace = [];
  for (const line of lines) {
    if (mode === null) {
      const m = line.match(/^@@ (\S+)(?: (\S+))?$/);
      if (m) {
        if (m[1] === "count") head.count = Number(m[2]);
        else head = { id: m[1] };
      } else if (line === "<<< find") {
        if (!head) throw new Error("proxy_patches.txt: find block without @@ id");
        mode = "find";
      }
    } else if (mode === "find") {
      if (line === "=== replace") mode = "replace";
      else find.push(line);
    } else if (mode === "replace") {
      if (line === ">>>") {
        out.push({ id: head.id, worker: "proxy", find: find.join("\n"), replace: replace.join("\n"), ...(head.count ? { count: head.count } : {}) });
        head = null; mode = null; find = []; replace = [];
      } else replace.push(line);
    }
  }
  if (mode !== null) throw new Error("proxy_patches.txt: unterminated block");
  return out;
}

export default readPatches();
