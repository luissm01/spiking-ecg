// Ejecuta el script REAL del HTML en Node; el DOM simulado solo evita dibujar.
// La instrumentación existe únicamente aquí, no expone estados en la página.
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname, '../docs/index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const context = new Proxy({}, {
  get: (_, key) => key === 'getContext' ? () => context : () => {},
  set: () => true,
});
const elements = new Map();
const document = {getElementById(id) {
  if (!elements.has(id)) elements.set(id, {
    width: 800, height: 210, style: {}, addEventListener() {},
    getContext: () => context,
  });
  return elements.get(id);
}};
const sandbox = {document, setInterval, clearInterval};
vm.createContext(sandbox);
const instrument = `
globalThis.snapshot = (label, retention, constantCurrent = false) => {
  model = freshModel(); sample = label; beta = retention;
  if (constantCurrent) { model.w.fill(0); model.b.fill(1); }
  const before = JSON.parse(JSON.stringify(model));
  const r = forward(), g = backward(r);
  forward();
  const repeated = forward();
  const stats = updateOnce();
  return {before, r, g, repeated, stats, after: model};
};
`;
vm.runInContext(script.replace('model=freshModel();render();',
  'model=freshModel();render();\n' + instrument), sandbox);
const cases = [];
for (const beta of [0.5, 0.9, 0.98]) {
  for (const sample of [0, 1]) cases.push(sandbox.snapshot(sample, beta));
}
const boundary = sandbox.snapshot(0, 0.9, true);
console.log(JSON.stringify({cases, boundary}));
