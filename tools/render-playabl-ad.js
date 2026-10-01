// Renders playabl-ad.html frame-by-frame. Run from the repo root:
//   node tools/render-playabl-ad.js video playabl-ad-silent.mp4 [fps]
//   node tools/render-playabl-ad.js events playabl-ad-events.json   (sound cues for playabl-ad-audio.js)
//   node tools/render-playabl-ad.js stills <outdir> 1000,5000,...   (PNG keyframes at those ms)
const { chromium } = require('playwright');
const { spawn } = require('child_process');
const fs = require('fs');

(async () => {
  const [mode, out, arg] = process.argv.slice(2);
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 1080, height: 1920 } });
  const errs = []; p.on('pageerror', e => errs.push(e.message));
  await p.goto('file://' + process.cwd() + '/playabl-ad.html?capture');
  await p.evaluate(() => window.AD.ready);

  if (mode === 'events') {
    fs.writeFileSync(out, JSON.stringify(await p.evaluate(() => AD.events()), null, 1));
  } else if (mode === 'stills') {
    fs.mkdirSync(out, { recursive: true });
    for (const ms of arg.split(',').map(Number)) {
      await p.evaluate(t => AD.render(t), ms);
      await p.screenshot({ path: `${out}/st-${ms}.png` });
    }
  } else {
    const fps = +(arg || 60);
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
  if (errs.length) { console.error('PAGE ERRORS', errs); process.exitCode = 1; }
  await b.close();
})();
