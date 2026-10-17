#!/bin/sh
# Builds docs/ for GitHub Pages from the same files that get published to claude.ai.
# claude.ai wraps index.html in its own <!doctype html> page; other hosts need it in the file,
# and there "Sao chép link" must copy the page's own address, not the private claude.ai artifact.
# Run after every change, then commit docs/ too:  ./build-pages.sh && git add -A && git commit
set -e
cd "$(dirname "$0")"
rm -rf docs
mkdir -p docs/models
cp blocks.js live.js docs/
cp models/*.js docs/models/
{
  printf '<!doctype html>\n<html lang="vi">\n'
  sed "s#^const ARTIFACT_URL = '[^']*';#const ARTIFACT_URL = '';#" index.html
  printf '\n</html>\n'
} > docs/index.html
grep -q "^const ARTIFACT_URL = '';" docs/index.html || { echo "build-pages.sh: ARTIFACT_URL line not found in index.html" >&2; exit 1; }
echo "docs/ ready ($(find docs -type f | wc -l | tr -d ' ') files)"
