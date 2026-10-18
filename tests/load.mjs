// Runs the page's classic scripts in one Node context, in the order index.html loads them.
// console.assert throws here, so the self-checks inside the scripts fail the tests instead of only logging.
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const ROOT = new URL('../', import.meta.url);

export function run(sources) {
  const assert = (ok, ...msg) => { if (!ok) throw new Error(`console.assert failed: ${msg.join(' ')}`); };
  const ctx = vm.createContext({ console: Object.assign(Object.create(console), { assert }) });
  for (const { name, code } of sources) vm.runInContext(code, ctx, { filename: name });
  return expr => vm.runInContext(expr, ctx); // top-level const/let are only reachable by evaluating their names
}

// the local <script src> tags of index.html (CDN scripts and the inline module are skipped)
export const pageScripts = () => [...readFileSync(new URL('index.html', ROOT), 'utf8').matchAll(/<script src="([^"]+)"><\/script>/g)]
  .map(m => m[1]).filter(src => !/^https?:/.test(src));

export const loadPage = () => run(pageScripts().map(name => ({ name, code: readFileSync(new URL(name, ROOT), 'utf8') })));
