#!/usr/bin/env bash
# Rebuilds playabl-ad.mp4 (video + soundtrack) from playabl-ad.html. Run from the repo root.
set -euo pipefail
tmp=$(mktemp -d)
node tools/render-playabl-ad.js events "$tmp/events.json"
node tools/playabl-ad-audio.js "$tmp/events.json" playabl-ad-audio.wav
node tools/render-playabl-ad.js video "$tmp/silent.mp4" 60
ffmpeg -loglevel error -y -i "$tmp/silent.mp4" -i playabl-ad-audio.wav -map 0:v -map 1:a -c:v copy \
  -c:a aac -b:a 192k -ar 48000 -af "loudnorm=I=-14:TP=-1.5:LRA=11" -shortest -movflags +faststart playabl-ad.mp4
rm -rf "$tmp"
echo "built playabl-ad.mp4"
