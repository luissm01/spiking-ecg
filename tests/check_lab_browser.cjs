// Smoke test opcional en Chromium real; Playwright se instala fuera del proyecto.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const {chromium} = require('playwright');

(async () => {
  const file = path.resolve(__dirname, '../docs/index.html');
  const server = http.createServer((req, res) => {
    res.writeHead(200, {'Content-Type': 'text/html; charset=utf-8'});
    res.end(fs.readFileSync(file));
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  let browser;
  try {
    browser = await chromium.launch({headless: true});
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const urls = [pathToFileURL(file).href,
      `http://127.0.0.1:${server.address().port}/spiking-ecg/`];
    for (const url of urls) {
      for (const width of [1280, 390, 320]) {
        await page.setViewportSize({width, height: 900});
        await page.goto(url);
        assert.equal(await page.locator('#tLabel').textContent(), '1/140');
        assert.equal(await page.locator('#bins .bin').count(), 7);
        const loss = await page.locator('#loss').textContent();
        await page.click('#next');
        assert.equal(await page.locator('#tLabel').textContent(), '2/140');
        assert.equal(await page.locator('#loss').textContent(), loss);
        await page.click('#rewind');
        await page.click('#play');
        await page.waitForFunction(() => document.getElementById('tLabel').textContent !== '1/140');
        await page.click('#play');
        await page.click('#rewind');
        await page.selectOption('#sample', '1');
        assert.equal(await page.locator('#truth').textContent(), 'Resto');
        await page.locator('#neuron').fill('31');
        await page.locator('#beta').fill('0.98');
        assert.equal(await page.locator('#neuronLabel').textContent(), '31');
        assert.equal(await page.locator('#betaLabel').textContent(), '0.98');
        const originalWeight = await page.locator('#wLabel').textContent();
        await page.click('#train');
        await page.click('#train10');
        assert.match(await page.locator('#trainingInfo').textContent(), /aprendizaje: 11/);
        assert.ok(Number.isFinite(Number(await page.locator('#loss').textContent())));
        await page.click('#reset');
        assert.equal(await page.locator('#wLabel').textContent(), originalWeight);
        assert.equal(await page.locator('#betaLabel').textContent(), '0.98');
        await page.selectOption('#sample', '0');
        assert.equal(await page.locator('#truth').textContent(), 'Normal');
        const overflow = await page.evaluate(() =>
          document.documentElement.scrollWidth > window.innerWidth);
        assert.equal(overflow, false, `Desbordamiento horizontal a ${width}px`);
      }
    }
    assert.deepEqual(errors, []);
    if (process.env.LAB_SCREENSHOT) {
      await page.setViewportSize({width: 1280, height: 900});
      await page.screenshot({path: process.env.LAB_SCREENSHOT, fullPage: true});
    }
    console.log('Chromium: file:// y HTTP bajo /spiking-ecg/, 1280/390/320px; controles y canvas sin errores.');
  } finally {
    if (browser) await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
