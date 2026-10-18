"""
Local web front end for the stereo reconstruction pipeline.

    python3 serve.py
    then open http://localhost:8000

Pick a left and a right image from your computer, optionally a calib.txt,
and get the reconstruction back in the browser.

NOTE ON THE ZERO-IMPORT RULE. This file is a *wrapper*, not the model.
It imports from the standard library so it can speak HTTP. The model
itself -- stereonet.py -- still has zero import statements, and this file
loads it as source text and executes it, so nothing here leaks into it.
Keeping the boundary explicit matters: the claim is that the computer
vision is written from scratch, not that a web server was.
"""

import html
import http.server
import io
import os
import socketserver
import sys
import threading
import time
import traceback
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "_uploads")
os.makedirs(WORK, exist_ok=True)

# ---------------------------------------------------------------- model
_src = open(os.path.join(HERE, "stereonet.py")).read().replace(
    "if __name__ == '__main__':\n    run_demo()\n", "")
SN = {"__name__": "stereonet_lib"}
exec(_src, SN)

WEIGHTS = os.path.join(HERE, "stereonet_weights.txt")

JOBS = {}
LOCK = threading.Lock()


class _Tee(io.TextIOBase):
    """Collect pipeline stdout so the browser can show live progress."""

    def __init__(self, job):
        self.job = job

    def write(self, s):
        if s.strip():
            with LOCK:
                self.job["log"].append(s.rstrip("\n"))
        return len(s)


def run_job(job_id, left, right, calib, quality):
    job = JOBS[job_id]
    presets = {
        "fast":     dict(display_width=288, refine_iters=40),
        "balanced": dict(display_width=400, refine_iters=90),
        "best":     dict(display_width=560, refine_iters=150),
    }
    opts = presets.get(quality, presets["balanced"])
    out_prefix = os.path.join(WORK, job_id)
    old = sys.stdout
    sys.stdout = _Tee(job)
    try:
        t0 = time.time()
        SN["demo_files"](left, right, out_prefix=out_prefix,
                         weights=WEIGHTS, calib=calib, **opts)
        print("done in %.1f s" % (time.time() - t0))
        job["result"] = out_prefix + "_viewer.html"
        job["status"] = "done"
    except Exception as exc:
        job["status"] = "error"
        job["error"] = "%s: %s" % (type(exc).__name__, exc)
        with LOCK:
            job["log"].append("ERROR " + job["error"])
            job["log"].append(traceback.format_exc().strip().split("\n")[-1])
    finally:
        sys.stdout = old


# ---------------------------------------------------------------- forms
def parse_multipart(body, boundary):
    """Minimal multipart/form-data parser.

    cgi.FieldStorage was removed in Python 3.13, and pulling in a web
    framework for one form would be absurd, so this walks the parts by
    hand. Returns {name: (filename, bytes)}.
    """
    out = {}
    sep = b"--" + boundary
    for part in body.split(sep):
        if not part or part in (b"--\r\n", b"--"):
            continue
        if b"\r\n\r\n" not in part:
            continue
        head, data = part.split(b"\r\n\r\n", 1)
        if data.endswith(b"\r\n"):
            data = data[:-2]
        headers = head.decode("utf-8", "replace")
        name = None
        filename = ""
        for token in headers.split(";"):
            token = token.strip()
            if token.startswith("name="):
                name = token[5:].strip('"').split("\r\n")[0].strip('"')
            elif token.startswith("filename="):
                filename = token[9:].strip('"').split("\r\n")[0].strip('"')
        if name:
            out[name] = (filename, data)
    return out


PAGE = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stereo 3D reconstruction</title><style>
:root{--bg:#0e1013;--panel:#171a1f;--line:#272c34;--ink:#e8eaed;
 --dim:#9aa3af;--accent:#5aa9ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);padding:20px;
 font:14px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:860px;margin:0 auto}
h1{font-size:21px;margin:0 0 3px}
.sub{color:var(--dim);margin-bottom:22px}
.card{background:var(--panel);border:1px solid var(--line);
 border-radius:11px;padding:18px;margin-bottom:14px}
label.f{display:block;margin-bottom:14px}
label.f span{display:block;font-size:12px;color:var(--dim);
 text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px}
input[type=file]{width:100%;background:#11141a;border:1px dashed var(--line);
 border-radius:8px;padding:13px;color:var(--ink);cursor:pointer}
input[type=file]::file-selector-button{background:#222831;color:var(--ink);
 border:1px solid var(--line);border-radius:6px;padding:6px 12px;
 margin-right:12px;cursor:pointer}
.row{display:flex;gap:14px;flex-wrap:wrap}
.row>*{flex:1 1 240px}
select{width:100%;background:#11141a;color:var(--ink);
 border:1px solid var(--line);border-radius:8px;padding:10px}
button{background:var(--accent);color:#06131f;border:0;border-radius:8px;
 padding:12px 22px;font-size:15px;font-weight:600;cursor:pointer}
button:disabled{opacity:.5;cursor:default}
#log{background:#0a0c0f;border:1px solid var(--line);border-radius:8px;
 padding:12px;font:12.5px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;
 color:#b7c0cc;white-space:pre-wrap;max-height:260px;overflow:auto;
 display:none}
.hint{color:var(--dim);font-size:12.5px;margin-top:10px}
.warn{color:#ffc46b}
iframe{width:100%;height:80vh;border:1px solid var(--line);
 border-radius:11px;background:#000;display:none;margin-top:14px}
</style></head><body><div class="wrap">
<h1>Stereo 3D reconstruction</h1>
<div class="sub">Two rectified images in, a reconstruction you can rotate out.</div>

<div class="card"><form id="f">
 <div class="row">
  <label class="f"><span>Left image</span>
   <input type="file" name="left" accept="image/png,image/x-portable-pixmap,image/x-portable-graymap,.png,.ppm,.pgm" required></label>
  <label class="f"><span>Right image</span>
   <input type="file" name="right" accept="image/png,image/x-portable-pixmap,image/x-portable-graymap,.png,.ppm,.pgm" required></label>
 </div>
 <div class="row">
  <label class="f"><span>calib.txt (optional)</span>
   <input type="file" name="calib" accept=".txt"></label>
  <label class="f"><span>Quality</span>
   <select name="quality">
    <option value="fast">Fast &mdash; about 30 s with PyPy</option>
    <option value="balanced" selected>Balanced &mdash; about 75 s with PyPy</option>
    <option value="best">Best &mdash; about 2.5 min with PyPy</option>
   </select></label>
 </div>
 <button id="go" type="submit">Reconstruct</button>
 <div class="hint">PNG, PGM or PPM. Both images must be the same size and
  <strong>rectified</strong> &mdash; the same point must fall on the same row
  in both. Benchmark pairs already are; two handheld photos are not.
  Without a calib.txt the geometry is correct up to an overall scale.</div>
</form></div>

<pre id="log"></pre>
<iframe id="out"></iframe>

<script>
const f=document.getElementById('f'), go=document.getElementById('go'),
      log=document.getElementById('log'), out=document.getElementById('out');
f.onsubmit=async e=>{
  e.preventDefault();
  go.disabled=true; go.textContent='Working\\u2026';
  out.style.display='none'; log.style.display='block';
  log.textContent='uploading\\u2026';
  const r=await fetch('/run',{method:'POST',body:new FormData(f)});
  const {id}=await r.json();
  const poll=setInterval(async()=>{
    const s=await (await fetch('/status?id='+id)).json();
    log.textContent=s.log.join('\\n');
    log.scrollTop=log.scrollHeight;
    if(s.status==='done'){
      clearInterval(poll);
      go.disabled=false; go.textContent='Reconstruct';
      out.src='/result?id='+id; out.style.display='block';
      out.scrollIntoView({behavior:'smooth'});
    } else if(s.status==='error'){
      clearInterval(poll);
      go.disabled=false; go.textContent='Reconstruct';
    }
  },1000);
};
</script></div></body></html>
"""


class Handler(http.server.BaseHTTPRequestHandler):
    def _send(self, code, ctype, body):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass                      # keep the console clean for the demo

    def do_GET(self):
        path = self.path.split("?")[0]
        query = {}
        if "?" in self.path:
            for kv in self.path.split("?", 1)[1].split("&"):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    query[k] = v

        if path == "/":
            return self._send(200, "text/html; charset=utf-8", PAGE)

        if path == "/status":
            job = JOBS.get(query.get("id", ""))
            if job is None:
                return self._send(404, "application/json", '{"status":"gone"}')
            with LOCK:
                body = ('{"status":"%s","log":[%s]}'
                        % (job["status"],
                           ",".join(['"%s"' % l.replace('\\', '\\\\')
                                                .replace('"', '\\"')
                                     for l in job["log"]])))
            return self._send(200, "application/json", body)

        if path == "/result":
            job = JOBS.get(query.get("id", ""))
            if job is None or job.get("status") != "done":
                return self._send(404, "text/plain", "not ready")
            return self._send(200, "text/html; charset=utf-8",
                              open(job["result"], "rb").read())

        self._send(404, "text/plain", "not found")

    def do_POST(self):
        if self.path != "/run":
            return self._send(404, "text/plain", "not found")
        ctype = self.headers.get("Content-Type", "")
        if "boundary=" not in ctype:
            return self._send(400, "text/plain", "expected multipart form")
        boundary = ctype.split("boundary=")[1].strip().strip('"').encode()
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        fields = parse_multipart(body, boundary)

        job_id = uuid.uuid4().hex[:12]
        paths = {}
        for key in ("left", "right", "calib"):
            fn, data = fields.get(key, ("", b""))
            if not fn or not data:
                paths[key] = None
                continue
            ext = os.path.splitext(fn)[1] or ".png"
            p = os.path.join(WORK, "%s_%s%s" % (job_id, key, ext))
            open(p, "wb").write(data)
            paths[key] = p

        if not paths["left"] or not paths["right"]:
            return self._send(400, "application/json",
                              '{"error":"need both images"}')

        quality = fields.get("quality", ("", b"balanced"))[1].decode()
        JOBS[job_id] = {"status": "running", "log": [], "result": None}
        threading.Thread(target=run_job, args=(job_id, paths["left"],
                                               paths["right"],
                                               paths["calib"], quality),
                         daemon=True).start()
        self._send(200, "application/json", '{"id":"%s"}' % job_id)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main(port=8000):
    if not os.path.exists(WEIGHTS):
        print("no weights at %s -- train first or copy the checkpoint in"
              % WEIGHTS)
        return
    cfg = SN["read_checkpoint_config"](WEIGHTS)
    print("stereo reconstruction server")
    print("  model:  %s" % (cfg if cfg else "legacy checkpoint"))
    print("  open:   http://localhost:%d" % port)
    print("  stop:   Ctrl-C")
    Server(("", port), Handler).serve_forever()


if __name__ == "__main__":
    p = 8000
    if len(sys.argv) > 1:
        p = int(sys.argv[1])
    main(p)
