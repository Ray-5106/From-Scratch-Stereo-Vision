// Compare the JavaScript refinement with the Python reference.
const R = require('./refine.js'), fs = require('fs');
const c = JSON.parse(fs.readFileSync(process.argv[2] || '/tmp/refine_ref.json'));
const t = Date.now();
const r = R.refineDisparity(Float64Array.from(c.left), Float64Array.from(c.right), Float64Array.from(c.rgb), Float64Array.from(c.disp0), c.W, c.H);
const ms = Date.now() - t;
let worst = 0, nbad = 0, stdiff = 0;
for (let i = 0; i < r.disp.length; i++) { const d = Math.abs(r.disp[i] - c.ref[i]); if (d > worst) worst = d; if (d > 1e-9) nbad++; if (r.status[i] !== c.status[i]) stdiff++; }
const mf = R.backgroundRegion(Float64Array.from(c.ref), c.W, c.H);
let mw = 0, sd = 0; for (let i = 0; i < mf.m.length; i++) { mw = Math.max(mw, Math.abs(mf.m[i] - c.bgm[i])); if (mf.src[i] !== c.src[i] || mf.mask[i] !== c.mask[i]) sd++; }
console.log(`${c.scene} ${c.W}x${c.H}: refinement in ${ms} ms; max |JS - Python| ${worst.toExponential(2)} px, ${nbad} pixels differ, ${stdiff} status differences; background layer max diff ${mw}, ${sd} mask/source differences`);
console.log(worst < 1e-6 && stdiff === 0 && mw === 0 && sd === 0 ? 'PASS' : 'FAIL');
