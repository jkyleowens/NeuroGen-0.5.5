const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

test('desktop entry points and renderer assets exist', () => {
  for (const file of ['electron/main.js','electron/preload.js','renderer/index.html','renderer/styles.css','renderer/app.js']) {
    assert.ok(fs.existsSync(path.join(__dirname, '..', file)), `${file} should exist`);
  }
});

test('renderer provides each core work utility', () => {
  const html = fs.readFileSync(path.join(__dirname, '..', 'renderer/index.html'), 'utf8');
  for (const utility of ['Projects', 'Inventory', 'Team', 'Time', 'Focus timer']) assert.match(html, new RegExp(utility));
});
