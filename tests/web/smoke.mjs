// Browser smoke test of the PWA in headless Chromium (desktop; no iPhone).
//
//   cd tests/web
//   npm install
//   npx playwright install chromium
//   node smoke.mjs [--photo path/to/face.jpg]
//
// Serves pwa/ on a local port, then checks: first-launch screen, model loading
// (WebAssembly), camera start with Chromium's fake camera (a test pattern, so no
// face is expected), optional photo analysis, settings reset, service worker and
// an offline reload. Exits with code 1 if a check fails or the page logs an error.

import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const PWA = resolve(fileURLToPath(new URL('../../pwa/', import.meta.url)));
const photoArg = process.argv.indexOf('--photo');
const PHOTO = photoArg > 0 ? resolve(process.argv[photoArg + 1]) : null;
const TYPES = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.json': 'application/json',
  '.webmanifest': 'application/manifest+json', '.wasm': 'application/wasm', '.onnx': 'application/octet-stream',
  '.png': 'image/png', '.svg': 'image/svg+xml', '.woff2': 'font/woff2',
};

const server = createServer(async (req, res) => {
  const path = normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname)).replace(/^([/\\])+/, '');
  const file = join(PWA, path === '' || path === '.' || path.endsWith('/') ? join(path, 'index.html') : path);
  try {
    const body = await readFile(file);
    res.writeHead(200, { 'Content-Type': TYPES[extname(file)] || 'application/octet-stream' });
    res.end(body);
  } catch {
    res.writeHead(404).end();
  }
});
await new Promise((r) => server.listen(0, r));
const URL_ = `http://localhost:${server.address().port}/`;

const results = [];
const check = (name, ok, detail = '') => {
  results.push(ok);
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  (${detail})` : ''}`);
};

const browser = await chromium.launch({ args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'] });
const context = await browser.newContext({ viewport: { width: 393, height: 852 }, permissions: ['camera'] });
const page = await context.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(String(e)));
page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
const state = () => page.evaluate(() => ({ view: document.body.dataset.view, live: document.body.dataset.live }));

try {
  await page.goto(URL_);
  check('first-launch screen shown', await page.isVisible('#onboarding'));
  await page.waitForFunction(() => /WASM|WEBGPU/.test(document.getElementById('engineChip').textContent), null, { timeout: 90000 });
  check('models loaded in the background', true, await page.textContent('#engineChip'));
  await page.click('#onbStart');
  await page.waitForFunction(() => document.getElementById('onboarding').hidden, null, { timeout: 30000 });
  check('Get started opens Live', (await state()).view === 'live');

  await page.click('#startCam');
  await page.waitForFunction(() => document.body.dataset.live === 'running', null, { timeout: 30000 });
  check('camera starts', true, await page.textContent('#faceCount'));

  if (PHOTO) {
    await page.click('.modes [data-view="photo"]');
    await page.setInputFiles('#photoPick', PHOTO);
    await page.waitForFunction(() => document.querySelectorAll('#faceList > *').length > 0 || /NO FACE/i.test(document.getElementById('photoMeta').textContent), null, { timeout: 60000 });
    const faces = await page.$$eval('#faceList > *', (e) => e.length);
    check('photo analysed', faces > 0, `${faces} face(s)`);
    await page.click('.modes [data-view="live"]');
  }

  await page.click('#openSettings');
  await page.selectOption('select[name="maxFaces"]', '3');
  const barShown = await page.isVisible('#resetBar');
  await page.click('#resetTop');
  check('settings reset', barShown && (await page.inputValue('select[name="maxFaces"]')) === '5');
  await page.click('#closeSettings');

  await page.waitForFunction(() => !!navigator.serviceWorker.controller, null, { timeout: 120000 }).catch(() => {});
  await page.reload();
  await page.waitForFunction(() => !!navigator.serviceWorker.controller, null, { timeout: 30000 }).catch(() => {});
  check('service worker active', await page.evaluate(() => !!navigator.serviceWorker.controller));
  await context.setOffline(true);
  await page.reload();
  await page.waitForFunction(() => /WASM|WEBGPU/.test(document.getElementById('engineChip').textContent), null, { timeout: 60000 }).catch(() => {});
  check('works offline (reload + models)', /WASM|WEBGPU/.test(await page.textContent('#engineChip')));
  await context.setOffline(false);
} catch (e) {
  check('test run', false, e.message.split('\n')[0]);
}

check('no errors in the page console', errors.length === 0, errors.slice(0, 3).join(' | '));
await browser.close();
server.close();
process.exit(results.every(Boolean) ? 0 : 1);
