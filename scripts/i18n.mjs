// The site's texts that need an English translation, and which ones i18n/en.js still lacks.
//   node scripts/i18n.mjs                 coverage per model
//   node scripts/i18n.mjs missing <model> the missing texts of one model (or "shared": concepts and papers), as entries
//                                         for i18n/en.js still holding the Vietnamese: translate each value in place
import { existsSync } from 'node:fs';
import { loadPage } from '../tests/load.mjs';

export const VI = /[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]/i; // text shown in Vietnamese

// every Vietnamese string a model shows: block names and tags, shapes, panel texts, formulas, code comments, link labels, sources
export function texts(root) {
  const out = new Set(), add = s => { if (typeof s === 'string' && VI.test(s)) out.add(s); };
  (function walk(nd) {
    ['name', 'tag', 'label', 'outLabel', 'hint', 'shape', 'lshape'].forEach(k => add(nd[k]));
    const i = nd.info || {};
    ['what', 'why', 'example', 'params', 'code'].forEach(k => add(i[k]));
    (i.formula || []).forEach(add);
    (nd.ref || []).forEach(r => { add(r.t); add(r.note); });
    [...(nd.links || []), ...(nd.extra || [])].forEach(l => add(l.label));
    (nd.skips || []).forEach(s => add(s[2]));
    (nd.children || []).forEach(walk);
  })(root);
  return [...out];
}

// texts not tied to one model: concept names ("Cũng xuất hiện ở") and paper titles
export const shared = get => [...Object.keys(get('CONCEPTS')), ...Object.values(get('PAPER')).map(p => p[0])].filter(s => VI.test(s));

export function load() {
  const get = loadPage(existsSync(new URL('../i18n/en.js', import.meta.url)) ? ['i18n/en.js'] : []);
  return { get, MODELS: get('MODELS'), textId: get('textId'), EN: get('typeof EN_TEXT === "undefined" ? {} : EN_TEXT') };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const { get, MODELS, textId, EN } = load(), [cmd, which] = process.argv.slice(2);
  const groups = { ...Object.fromEntries(Object.entries(MODELS).map(([k, m]) => [k, texts(m)])), shared: shared(get) };
  if (cmd === 'missing') {
    const seen = new Set();
    for (const s of groups[which]) {
      const id = textId(s);
      if (EN[id] || seen.has(id)) continue;
      seen.add(id);
      const short = s.replace(/\s+/g, ' '), note = short.length > 70 ? short.slice(0, 69) + '…' : short;
      console.log(`  ${JSON.stringify(id)}: ${JSON.stringify(s)}, // ${note}`);
    }
  } else {
    for (const [k, list] of Object.entries(groups)) {
      const done = list.filter(s => EN[textId(s)]).length;
      console.log(`${k.padEnd(13)} ${String(done).padStart(4)}/${String(list.length).padEnd(4)} ${Math.round(100 * done / Math.max(1, list.length))}%`);
    }
  }
}
