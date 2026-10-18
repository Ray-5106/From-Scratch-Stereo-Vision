// Compare the JavaScript port against the Python reference outputs.
const S = require('./stereonet_infer.js');
const fs = require('fs');
const ckpt = JSON.parse(fs.readFileSync(process.argv[2]));
const cases = JSON.parse(fs.readFileSync(process.argv[3]));
const net = S.buildNet(ckpt);
let worstAll = 0;
for (const c of cases) {
  const L = Float64Array.from(c.left), R = Float64Array.from(c.right);
  const t = Date.now();
  const out = c.name.includes('predict_auto')
    ? S.predictAuto(net, L, R, c.W, c.H, { targetPeak: 34 }).disp
    : S.predict(net, L, R, c.W, c.H);
  const ms = Date.now() - t;
  let worst = 0, sum = 0;
  for (let i = 0; i < out.length; i++) {
    const d = Math.abs(out[i] - c.ref[i]); sum += d; if (d > worst) worst = d;
  }
  worstAll = Math.max(worstAll, worst);
  console.log(`${c.name.padEnd(62)} max |JS - Python| ${worst.toExponential(2)}  mean ${(sum / out.length).toExponential(2)}  (${ms} ms in JS)`);
}
console.log(worstAll < 1e-6 ? `PASS: JavaScript port matches Python, worst difference ${worstAll.toExponential(2)} px`
                            : `FAIL: worst difference ${worstAll} px`);
