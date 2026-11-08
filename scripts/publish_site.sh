#!/usr/bin/env bash
# Publish the replay site to GitHub Pages: the page in site/ plus the recordings
# made by `python -m smartfly.web.record` (runs/pages/replays).
#
#   scripts/publish_site.sh
#
# The gh-pages branch holds a single commit, replaced each time, so old
# recordings do not pile up in the repository.
set -euo pipefail
cd "$(dirname "$0")/.."
REPLAYS="${1:-runs/pages/replays}"
[[ -f "$REPLAYS/index.json" ]] || { echo "no recordings in $REPLAYS; run python -m smartfly.web.record" >&2; exit 1; }

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cp -R site/. "$tmp"
cp -R "$REPLAYS" "$tmp/replays"
perl -pi -e 's|<head>|<head>\n<meta name="smartfly-mode" content="replay">|' "$tmp/index.html"
touch "$tmp/.nojekyll"

git -C "$tmp" init -q -b gh-pages
git -C "$tmp" add -A
git -C "$tmp" -c user.name="$(git config user.name)" -c user.email="$(git config user.email)" \
  commit -q -m "Replay site, $(date +%Y-%m-%d)"
git -C "$tmp" push -q --force "$(git remote get-url origin)" gh-pages
echo "published $(du -sh "$tmp" | cut -f1) to the gh-pages branch"
