// Renders playabl-ad.html frame-by-frame to a smooth video. Run from the repo root:
//   node tools/render-playabl-ad.js playabl-ad.mp4 60   (or --stills for keyframe PNGs)
const { chromium } = require('playwright');
const { spawn } = require('child_process');
(async () => {
  const out = process.argv[2], fps = +(process.argv[3] || 60);
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 1080, height: 1920 } });
  const errs = []; p.on('pageerror', e => errs.push(e.message));
  await p.goto('file://' + process.cwd() + '/playabl-ad.html?capture');
  await p.evaluate(() => window.AD.ready);
  if (out === '--stills') {
    for (const ms of [1500, 4800, 8000, 10900, 13900, 16600, 18600, 21500]) {
      await p.evaluate(t => AD.render(t), ms);
      await p.screenshot({ path: `${"."}/st-${ms}.png` });
    }
  } else {
    const dur = await p.evaluate(() => AD.DURATION), n = Math.round(dur / 1000 * fps);
    const ff = spawn('ffmpeg', ['-loglevel', 'error', '-y', '-f', 'image2pipe', '-framerate', String(fps), '-c:v', 'mjpeg', '-i', '-',
      '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out], { stdio: ['pipe', 'inherit', 'inherit'] });
    for (let f = 0; f < n; f++) {
      await p.evaluate(t => AD.render(t), f * 1000 / fps);
      const buf = await p.screenshot({ type: 'jpeg', quality: 95 });
      if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once('drain', r));
      if (f % 300 === 0) console.log('frame', f, '/', n);
    }
    ff.stdin.end(); await new Promise(r => ff.on('close', r));
  }
  if (errs.length) console.log('PAGE ERRORS', errs);
  await b.close();
})();
