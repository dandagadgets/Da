# Animated Walima invitation

`make_video.py` turns a static invitation card into a ~32 second animated video
with music, ready to share on WhatsApp.

What happens in the video:

- The card starts blank and every line is revealed in reading order: the
  Bismillah writes itself right to left, the names and *Walima* are written in
  with a glowing gold pen, the other lines fade up, the ornaments open out from
  the centre. The lettering is the card's own pixels, so fonts and spelling
  match the original exactly.
- A slow camera move pushes in on the text and pulls back to the full card at
  the end.
- The lantern candles flicker. Rose petals and gold dust drift through the
  frame, and light glints run along the gold trim.
- A soft harp and music-box soundtrack plays, with chimes on the key reveals.
  It is synthesized in the script, so there are no copyright issues.

## Usage

```bash
pip install numpy scipy pillow opencv-python-headless   # plus ffmpeg on PATH
python3 make_video.py                      # walima_invitation.mp4        1080x1620 (card shape)
python3 make_video.py --format story       # walima_invitation_story.mp4  1080x1920 (9:16, Status/Reels)
python3 make_video.py --hq                 # walima_invitation_hq.mp4     1440x2160 sharpened master, higher bitrate
python3 make_video.py --no-audio           # same video, no music
python3 make_video.py --stills 5,12,28     # quick PNG previews of single moments
```

The script expects the card image as `card.jpg` next to it (or pass
`--card path/to/card.jpg`). The layout table (`ELEMENTS`) at the top of the
script holds each line's position on the card, its reveal style, its start
time and its duration. Edit that table to change the pacing.

The card image and rendered videos are git-ignored here because they contain
personal details (names, phone numbers, venue) and this repository is public.
