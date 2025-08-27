# app.py
# PHPAiModel-RNN — Python port of generator_weights.php
# - Minimal UI (Flask)
# - Streams training logs (progress %, ETA, avg loss)
# - Simple tanh-RNN with Adagrad on tokenized RU/EN text datasets
# - Saves model JSON to ./Models
#
# GPU: CuPy (CUDA 11.x) if available, otherwise CPU (NumPy)
# Author: ported for Artur Strazewicz project
# License: MIT (2025)

import os, re, json, time, math
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as _np  # always have NumPy around
from flask import Flask, Response, request, render_template_string, send_from_directory, stream_with_context

# --------- Backend (CuPy preflight with graceful fallback) ----------
_GPU_AVAILABLE = False
GPU_INIT_ERROR = ""
try:
    import cupy as xp  # try GPU arrays
    try:
        # quick preflight: matmul (hits cuBLAS) + device info
        _ = (xp.ones((2,2), dtype=xp.float32) @ xp.ones((2,2), dtype=xp.float32)).sum()
        import cupy as _cp
        _CUDA_RUNTIME = None
        _GPU_NAME = None
        try:
            from cupy.cuda import runtime, Device
            _CUDA_RUNTIME = runtime.runtimeGetVersion()
            _GPU_NAME = Device(0).attributes.get('Name') if hasattr(Device(0), 'attributes') else None
            # fallback way to get name
            if not _GPU_NAME:
                _GPU_NAME = Device(0).name if hasattr(Device(0), 'name') else None
        except Exception:
            pass
        _GPU_AVAILABLE = True
    except Exception as e:
        import numpy as xp  # fallback to CPU
        GPU_INIT_ERROR = f"GPU disabled: CuPy preflight failed: {e}"
        _GPU_AVAILABLE = False
except Exception as e:
    import numpy as xp
    GPU_INIT_ERROR = f"GPU disabled: CuPy import failed: {e}"
    _GPU_AVAILABLE = False

# --------------- Config ---------------
ROOT = Path(__file__).resolve().parent
DATASETS_DIR = ROOT / "Datasets"
MODELS_DIR   = ROOT / "Models"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --------------- Utils ---------------
def format_hms(sec: float) -> str:
    sec = max(0, int(round(sec)))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

def build_tokens(text: str) -> List[str]:
    text = text.replace("\r", "")
    text = re.sub(r"([\.\,\!\?\:\;\-])", r" \1 ", text)
    text = text.replace("\n", " <NL> ").strip()
    text = f"<BOS> {text} <EOS>"
    text = re.sub(r"\s+", " ", text)
    return text.split(" ")

def build_vocab(tokens: List[str]) -> Tuple[Dict[str,int], List[str]]:
    tok2id: Dict[str,int] = {}
    id2tok: List[str] = []
    for t in tokens:
        if t not in tok2id:
            tok2id[t] = len(id2tok)
            id2tok.append(t)
    return tok2id, id2tok

def to_cpu(a):
    """Return NumPy array for JSON export regardless of backend."""
    try:
        import cupy as cp
        if isinstance(a, cp.ndarray):
            return cp.asnumpy(a)
    except Exception:
        pass
    return _np.asarray(a)

# --------------- RNN Core ---------------
class TinyRNN:
    """
    Simple single-layer tanh RNN language model with softmax output.
    Shapes:
      Wxh: (H, V)
      Whh: (H, H)
      Why: (V, H)
      bh:  (H,)
      by:  (V,)
    """
    def __init__(self, V: int, H: int, scale: float = 0.05, seed: int = 42):
        rng = _np.random.default_rng(seed)  # RNG stays NumPy for reproducibility
        self.V, self.H = V, H
        # All tensors on xp backend (CuPy if GPU ok, else NumPy), float32
        self.Wxh = xp.asarray(((rng.random((H, V)) * 2 - 1) * scale), dtype=xp.float32)
        self.Whh = xp.asarray(((rng.random((H, H)) * 2 - 1) * scale), dtype=xp.float32)
        self.Why = xp.asarray(((rng.random((V, H)) * 2 - 1) * scale), dtype=xp.float32)
        self.bh  = xp.zeros((H,), dtype=xp.float32)
        self.by  = xp.zeros((V,), dtype=xp.float32)
        # Adagrad accumulators
        self.mWxh = xp.zeros_like(self.Wxh)
        self.mWhh = xp.zeros_like(self.Whh)
        self.mWhy = xp.zeros_like(self.Why)
        self.mbh  = xp.zeros_like(self.bh)
        self.mby  = xp.zeros_like(self.by)

    @staticmethod
    def softmax(z):
        z = z - z.max()
        e = xp.exp(z)
        return e / e.sum()

    def bptt(self, inputs: List[int], targets: List[int], hprev):
        """
        Forward + backward through a sequence chunk.
        Returns: loss, grads dict, last hidden state
        """
        H = self.H
        hs = []
        ps = []
        loss = 0.0
        hs_prev = hprev

        # Forward (no one-hot: take Wxh[:, idx])
        for idx_in, idx_tgt in zip(inputs, targets):
            a = self.Wxh[:, idx_in] + self.Whh @ hs_prev + self.bh
            h_t = xp.tanh(a)
            y_t = self.Why @ h_t + self.by
            p_t = self.softmax(y_t)
            loss -= float(xp.log(xp.maximum(p_t[idx_tgt], 1e-12)))
            hs.append(h_t)
            ps.append(p_t)
            hs_prev = h_t

        # Grads init
        dWxh = xp.zeros_like(self.Wxh)
        dWhh = xp.zeros_like(self.Whh)
        dWhy = xp.zeros_like(self.Why)
        dbh  = xp.zeros_like(self.bh)
        dby  = xp.zeros_like(self.by)
        dh_next = xp.zeros((H,), dtype=xp.float32)

        # Backward
        for t in range(len(inputs)-1, -1, -1):
            dy = ps[t].copy()
            dy[targets[t]] -= 1.0  # ∂L/∂y
            dWhy += xp.outer(dy, hs[t])
            dby  += dy

            dh = self.Why.T @ dy + dh_next
            dhraw = (1.0 - hs[t] * hs[t]) * dh  # tanh'
            dbh  += dhraw
            # Column update instead of outer with one-hot
            dWxh[:, inputs[t]] += dhraw

            h_prev_t = hs[t-1] if t > 0 else hprev
            dWhh += xp.outer(dhraw, h_prev_t)

            dh_next = self.Whh.T @ dhraw

        # clip
        clip = 5.0
        for G in (dWxh, dWhh, dWhy):
            xp.clip(G, -clip, clip, out=G)
        xp.clip(dbh, -clip, clip, out=dbh)
        xp.clip(dby, -clip, clip, out=dby)

        return loss, {"dWxh": dWxh, "dWhh": dWhh, "dWhy": dWhy, "dbh": dbh, "dby": dby}, hs[-1].copy()

    def step_adagrad(self, grads, lr: float):
        eps = 1e-8
        self.mWxh += grads["dWxh"] ** 2
        self.mWhh += grads["dWhh"] ** 2
        self.mWhy += grads["dWhy"] ** 2
        self.mbh  += grads["dbh"]  ** 2
        self.mby  += grads["dby"]  ** 2

        self.Wxh -= lr * grads["dWxh"] / (xp.sqrt(self.mWxh) + eps)
        self.Whh -= lr * grads["dWhh"] / (xp.sqrt(self.mWhh) + eps)
        self.Why -= lr * grads["dWhy"] / (xp.sqrt(self.mWhy) + eps)
        self.bh  -= lr * grads["dbh"]  / (xp.sqrt(self.mbh)  + eps)
        self.by  -= lr * grads["dby"]  / (xp.sqrt(self.mby)  + eps)

# --------------- Training loop (streaming) ---------------
def train_stream(dataset_file: str, H: int, SEQ: int, EPOCHS: int, LR: float):
    ds_path = DATASETS_DIR / dataset_file
    if not ds_path.is_file():
        yield f"Dataset not found: {dataset_file}\n"
        return

    t0 = time.time()
    text = ds_path.read_text(encoding="utf-8", errors="ignore")
    tokens = build_tokens(text)
    vocab, ivocab = build_vocab(tokens)
    V = len(vocab)
    ids = _np.array([vocab[t] for t in tokens], dtype=_np.int32)  # CPU indices

    steps_per_epoch = int(max(0, len(ids) - SEQ) // max(1, SEQ))
    total_steps = max(1, steps_per_epoch * EPOCHS)
    done_steps = 0

    backend = "GPU (CuPy/CUDA 13.x)" if _GPU_AVAILABLE else "CPU (NumPy)"
    if not _GPU_AVAILABLE and GPU_INIT_ERROR:
        yield GPU_INIT_ERROR + "\n"
    yield f"Backend: {backend}\n"
    if _GPU_AVAILABLE:
        try:
            import cupy as cp
            from cupy.cuda import Device, runtime
            dev = Device(0)
            name = getattr(dev, "name", None) or getattr(dev, "attributes", {}).get("Name")
            cudart = None
            try:
                cudart = runtime.runtimeGetVersion()
            except Exception:
                pass
            yield f"GPU: {name or 'Unknown'} | CUDA runtime: {cudart or 'n/a'}\n"
        except Exception:
            pass

    yield f"Dataset: {dataset_file}\nTokens: {len(tokens)}\nVocab: {V}\nH: {H}  SEQ: {SEQ}  Epochs: {EPOCHS}  LR: {LR}\n"
    yield f"Planned steps: {total_steps} (≈ {steps_per_epoch} / epoch)\n\n"

    rnn = TinyRNN(V=V, H=H, scale=0.05, seed=123)
    hprev = xp.zeros((H,), dtype=xp.float32)

    for epoch in range(1, EPOCHS + 1):
        loss_sum = 0.0
        nsteps = 0

        # walk in non-overlapping SEQ chunks (like PHP)
        for pos in range(0, len(ids) - SEQ, SEQ):
            inputs  = ids[pos:pos + SEQ].tolist()
            targets = ids[pos + 1:pos + SEQ + 1].tolist()

            loss, grads, hprev = rnn.bptt(inputs, targets, hprev)
            rnn.step_adagrad(grads, LR)
            loss_sum += loss
            nsteps += 1
            done_steps += 1

            if (done_steps % 50) == 0 or done_steps == total_steps:
                elapsed = time.time() - t0
                rate = (elapsed / done_steps) if done_steps > 0 else 0.0
                remain = max(0.0, (total_steps - done_steps) * rate)
                percent = round((done_steps / total_steps) * 100.0, 2)
                eta = format_hms(remain)
                spent = format_hms(elapsed)
                avg_loss = (loss_sum / nsteps) if nsteps else 0.0
                yield f"Progress: {percent:6.2f}% | ETA {eta} | Spent {spent} | epoch {epoch}/{EPOCHS} | step {nsteps}/{steps_per_epoch} | avg loss {avg_loss:.5f}\n"

        avg = (loss_sum / nsteps) if nsteps else 0.0
        yield f"Epoch {epoch}/{EPOCHS} done | avg loss={avg:.5f} | steps={nsteps}\n\n"

    # save model
    fname = f"rnn_{ds_path.stem}_H{H}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out = {
        "V": V, "H": H,
        "vocab": vocab,          # token -> id
        "ivocab": ivocab,        # id -> token
        "Wxh": to_cpu(rnn.Wxh).tolist(),
        "Whh": to_cpu(rnn.Whh).tolist(),
        "Why": to_cpu(rnn.Why).tolist(),
        "bh":  to_cpu(rnn.bh ).tolist(),
        "by":  to_cpu(rnn.by ).tolist(),
        "meta": {
            "dataset": dataset_file,
            "epochs": EPOCHS,
            "seq": SEQ,
            "lr": LR,
            "time": datetime.utcnow().isoformat() + "Z",
            "train_seconds": round(time.time() - t0, 3),
            "tokens": len(tokens),
            "backend": backend,
            "dtype": "float32"
        }
    }
    (MODELS_DIR / fname).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")

    total_spent = format_hms(time.time() - t0)
    yield f"DONE 100.00% | Total time {total_spent}\n"
    yield f"Saved: Models/{fname}\n"

# --------------- Web UI ---------------
HTML = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>RNN Trainer — Python</title>
<style>
:root{ --bg:#f7f8fb; --card:#ffffff; --text:#0f172a; --muted:#6b7280; --line:#e5e7eb; --accent:#2563eb; }
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font-family:System-ui,-apple-system,Segoe UI,Roboto,Inter,Arial}
header{background:var(--card);border-bottom:1px solid var(--line);padding:12px 16px}
main{max-width:980px;margin:0 auto;padding:16px}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
select,input,button{border:1px solid var(--line);border-radius:10px;padding:8px 10px;font:inherit;background:#fff;color:var(--text)}
button{background:var(--accent);color:#fff;border-color:transparent;cursor:pointer}
pre{white-space:pre-wrap;background:#fff;border:1px solid var(--line);border-radius:10px;padding:10px;min-height:200px;max-height:60vh;overflow:auto}
table{width:100%;border-collapse:collapse;margin-top:12px}
th,td{border-bottom:1px solid var(--line);padding:8px;text-align:left;font-size:14px}
.hint{font-size:12px;color:var(--muted)}
a{color:#4ea3ff;text-decoration:none}
</style>
</head>
<body>
<header>
    <b>RNN Trainer (Python)</b> — интерфейс обучения. После сохранения модели подключите её в своём чате.
</header>
<main>
<section>
    <div class="row">
        <label>Dataset</label>
        <select id="dataset">
            {% for f in datasets %}
            <option>{{f}}</option>
            {% endfor %}
        </select>
        <label>Hidden</label><input id="hidden" type="number" value="64" min="8" max="256" style="width:90px">
        <label>SeqLen</label><input id="seq" type="number" value="16" min="4" max="64" style="width:90px">
        <label>Epochs</label><input id="epochs" type="number" value="5" min="1" max="100" style="width:90px">
        <label>LR</label><input id="lr" type="number" step="0.001" value="0.05" style="width:90px">
        <button id="run">Training</button>
    </div>
    <p class="hint">Тренировка запускается сервером (Python/Flask). Логи (проценты, ETA, потеря) выводятся ниже в реальном времени. Файл модели сохраняется в <code>/Models</code>.</p>
    <pre id="log">Готов к обучению…</pre>
</section>

<section>
    <h3>Доступные модели</h3>
    <table>
        <thead><tr><th>Файл</th><th>Размер</th><th>Дата</th></tr></thead>
        <tbody>
        {% for m in models %}
        <tr>
            <td>{{m.name}}</td>
            <td>{{m.size}}</td>
            <td>{{m.mtime}}</td>
        </tr>
        {% endfor %}
        </tbody>
    </table>
</section>
</main>
<footer style="background:#222; color:#eee; text-align:center; padding:20px; font-family:Arial, sans-serif; font-size:14px;">
    <div style="margin-bottom:10px;">
        <strong>PHPAiModel-RNN</strong> © 2025 — MIT License
    </div>
    <div style="margin-bottom:10px;">
        Developed by <a href="https://www.linkedin.com/in/arthur-stark/">Artur Strazewicz</a>
    </div>
    <div>
        <a href="https://github.com/iStark/PHPAiModel-RNN">GitHub</a> |
        <a href="https://x.com/strazewicz">X (Twitter)</a> |
        <a href="https://truthsocial.com/@strazewicz">TruthSocial</a>
    </div>
</footer>
<script>
async function run(){
    const params = new URLSearchParams({
        dataset: document.getElementById('dataset').value,
        hidden:  document.getElementById('hidden').value,
        seq:     document.getElementById('seq').value,
        epochs:  document.getElementById('epochs').value,
        lr:      document.getElementById('lr').value
    });
    const url = '/run?' + params.toString();

    const logEl = document.getElementById('log');
    logEl.textContent = 'Запуск обучения…\\n';
    try {
        const res = await fetch(url, {cache:'no-store'});
        if (!res.body) {
            const txt = await res.text();
            logEl.textContent += txt + '\\n';
            return;
        }
        const reader = res.body.getReader();
        const decoder = new TextDecoder('utf-8');
        while (true) {
            const {done, value} = await reader.read();
            if (done) break;
            logEl.textContent += decoder.decode(value, {stream:true});
            logEl.scrollTop = logEl.scrollHeight;
        }
    } catch (e) {
        logEl.textContent += '\\nОшибка: ' + (e && e.message ? e.message : e) + '\\n';
    }
    setTimeout(()=>location.reload(), 600);
}
document.getElementById('run').onclick = run;
</script>
</body>
</html>
"""

app = Flask(__name__)

def list_datasets() -> List[str]:
    # показываем только .txt (удобнее переименовать файлы на *.txt)
    return sorted([f.name for f in DATASETS_DIR.glob("*.txt")])

def list_models():
    rows = []
    for p in sorted(MODELS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        sz = f"{p.stat().st_size:,}".replace(",", " ")
        mt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime))
        rows.append({"name": p.name, "size": f"{sz} B", "mtime": mt})
    return rows

@app.route("/")
def index():
    return render_template_string(HTML, datasets=list_datasets(), models=list_models())

@app.route("/run")
def run():
    dataset = request.args.get("dataset", "greetings_ru_en.txt")
    H      = max(8, int(request.args.get("hidden", 256)))
    SEQ    = max(4, int(request.args.get("seq", 32)))
    EPOCHS = max(1, int(request.args.get("epochs", 80)))
    LR     = max(0.0001, float(request.args.get("lr", 0.05)))

    @stream_with_context
    def generate():
        for chunk in train_stream(dataset, H, SEQ, EPOCHS, LR):
            yield chunk

    headers = {
        "Content-Type": "text/plain; charset=utf-8",
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
    }
    return Response(generate(), headers=headers)

@app.route("/Models/<path:filename>")
def get_model(filename):
    return send_from_directory(MODELS_DIR, filename, as_attachment=True)

if __name__ == "__main__":
    print("Starting RNN Trainer (Python) at http://127.0.0.1:5000/")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
