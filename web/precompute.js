const S = require('./stereonet_infer.js'), Rf = require('./refine.js'), fs = require('fs');
const net = S.buildNet(JSON.parse(fs.readFileSync(process.argv[2])));
const T = parseFloat(process.argv[3]);
const cases = JSON.parse(fs.readFileSync(process.argv[4])), out = {};
for (const c of cases) {
  const t = Date.now();
  const L = Float64Array.from(c.left), R = Float64Array.from(c.right);
  const r = S.predictAuto(net, L, R, c.W, c.H, { targetPeak: T });
  const t2 = Date.now();
  const ref = Rf.refineDisparity(L, R, Float64Array.from(c.rgb), r.disp, c.W, c.H);
  out[c.key] = { raw: Array.from(r.disp), ref: Array.from(ref.disp) };
  console.error(`  ${c.key}: ${c.W}x${c.H}, network at ${r.Ww}x${r.Hw} ${t2 - t} ms, refinement ${Date.now() - t2} ms`);
}
fs.writeFileSync(process.argv[5], JSON.stringify(out));
