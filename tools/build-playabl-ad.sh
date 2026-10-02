#!/usr/bin/env bash
# Rebuilds an ad's MP4 (video + soundtrack) from its HTML page. Run from the repo root:
#   tools/build-playabl-ad.sh                 -> playabl-ad.html   -> playabl-ad.mp4
#   tools/build-playabl-ad.sh playabl-ad-2    -> playabl-ad-2.html -> playabl-ad-2.mp4
#   tools/build-playabl-ad.sh playabl-clean playabl-clean-2 edit=2   (page, output name, query)
set -euo pipefail
name=${1:-playabl-ad}
out=${2:-$name}
export AD_PAGE="$name.html${3:+?$3}"
tmp=$(mktemp -d)
node tools/render-playabl-ad.js events "$tmp/events.json"
node tools/playabl-ad-audio.js "$tmp/events.json" "$out-audio.wav"
node tools/render-playabl-ad.js video "$tmp/silent.mp4" 60
ffmpeg -loglevel error -y -i "$tmp/silent.mp4" -i "$out-audio.wav" -map 0:v -map 1:a -c:v copy \
  -c:a aac -b:a 192k -ar 48000 -af "loudnorm=I=-14:TP=-1.5:LRA=11" -shortest -movflags +faststart "$out.mp4"
rm -rf "$tmp"
echo "built $out.mp4"
