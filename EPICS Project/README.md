# Breast Cancer Prediction & Prevention

A Flask web app that serves an awareness site, a scan **history/report** area, and an
image-based **risk estimator** built with transfer learning on EfficientNetB3.

> **Not a medical device.** The estimator is a research-grade statistical model. It does
> not diagnose breast cancer and must never replace examination, imaging or biopsy by a
> qualified clinician.

---

## How it works

```
browser ──▶ Flask (backend.py)
              ├─ auth: Flask-Login session, pbkdf2:sha256 password hashes
              ├─ data: Flask-SQLAlchemy → SQLite/Postgres (User, Scan)
              └─ /analyzer: upload → EfficientNetB3 → probability → risk tier
                                                  │
                                                  └─▶ Scan row (per user) → /report/<id>
```

1. A visitor lands on `/` (public awareness page) and signs up at `/signup`.
2. Logged-in users get `/dashboard`, `/analyzer` (detection), `/history`, `/report/<id>`.
   Public pages are `/`, `/login`, `/signup`, `/prevention`, `/symptoms`.
3. `POST /analyzer` validates the upload, scores it with the model, and maps the
   probability to a risk tier:

   | Probability | Label | Class |
   |---|---|---|
   | `< 0.30` | LOW PROBABILITY | `low-risk` |
   | `0.30 – 0.50` | MODERATE CONCERN | `mod-risk` |
   | `≥ 0.50` | HIGH PROBABILITY | `high-risk` |

4. Each analysis is stored as a `Scan` row owned by the current user. `/history` lists only
   that user's scans, and `/report/<id>` returns **403** for anyone else's scan.
5. Every page extends `templates/base.html` (shared nav, flash messages, mobile menu).

### Model

Two models were built in `Breast_Cancer_Prediction.ipynb`:

| Model | Test accuracy |
|---|---|
| Custom CNN from scratch | 71.14% — collapsed on the minority (cancer) class |
| **EfficientNetB3 transfer learning (deployed)** | **93.96%** |

The deployed model freezes ImageNet-pretrained EfficientNetB3 and adds augmented inputs,
global average pooling, dropout and a single sigmoid output — see `build_model()` in
`backend.py`. Weights load **lazily on first prediction** from `MODEL_PATH`.

Recorded metrics live in `model_metrics.json` (from the notebook's validation split):
accuracy 93.96%, cancer precision 94.44%, **cancer recall 68.00%**, cancer F1 79.07%,
non-cancer recall 99.19% (149 images, 25 cancer). The result page renders these figures.
Cancer recall is the weakest and most clinically important number: at a 0.5 threshold the
model misses roughly one in three cancers in that split.

To measure it properly (including ROC-AUC, which the notebook never recorded):

```bash
python evaluate.py --cancer-dir data/test/cancer --non-cancer-dir data/test/non_cancer
```

This rewrites `model_metrics.json`, and the app picks the new numbers up on the next
request.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file (never commit it):

```ini
SECRET_KEY=<run: python -c "import secrets; print(secrets.token_hex(32))">
DATABASE_URL=sqlite:///instance/app.db
MODEL_PATH=imagenet_model.h5
```

`imagenet_model.h5` (44 MB) is **not in version control** (see `.gitignore`). Download it
from the training notebook's saved location, or retrain with
`Breast_Cancer_Prediction.ipynb`, and point `MODEL_PATH` at it.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `SECRET_KEY` | random per boot | Session signing. **Set this in production.** |
| `DATABASE_URL` | `sqlite:///instance/app.db` | SQLAlchemy connection string |
| `MODEL_PATH` | `./imagenet_model.h5` | Trained weights |
| `MODEL_METRICS_PATH` | `./model_metrics.json` | Metrics shown on the result page |
| `UPLOAD_FOLDER` | `./uploads` | Temporary dir for in-flight scans |
| `MAX_UPLOAD_BYTES` | `5242880` (5 MB) | Max request body size |
| `SESSION_COOKIE_SECURE` | `0` | Set to `1` behind HTTPS |
| `CSRF_ENABLED` | `1` | Session-token CSRF checks on POST |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | Dev server bind |
| `FLASK_DEBUG` | `0` | Set to `1` for the reloader/debug traces |

## Running

```bash
python backend.py            # http://127.0.0.1:8000
```

Tables are created on startup (`init_db()`). For production, run a real WSGI server
behind HTTPS with `SECRET_KEY`, `SESSION_COOKIE_SECURE=1` and `FLASK_DEBUG=0` set —
on Windows use `waitress-serve --port=8000 backend:app`, on Linux
`gunicorn -w 2 -b 127.0.0.1:8000 backend:app`. Note the first prediction triggers the
model load (several seconds) — warm it up or give the worker a generous timeout.

## Tests

```bash
python -m pytest tests/ -q
```

44 tests cover public/protected routing, signup validation, login/logout, CSRF
enforcement, the risk-tier boundaries, upload validation (missing file, bad extension,
oversized body), scan persistence, per-user isolation of reports (403/404), and the
metrics panel. The Keras model is stubbed, so the suite runs without the weights file.

## Security notes

- Passwords are stored as `pbkdf2:sha256` hashes via Werkzeug.
- CSRF: every POST carries a session token, verified in `before_request` with
  `hmac.compare_digest`. `abort(400)` fires before the view runs.
- Uploads are extension-checked, size-capped, written under a random filename in
  `UPLOAD_FOLDER`, and **deleted immediately after scoring** (`finally` block).
- Session cookies are `HttpOnly` + `SameSite=Lax`; set `SESSION_COOKIE_SECURE=1` on HTTPS.
- Errors return the `error.html` page; raw exception text is logged server-side rather
  than echoed to the browser.

## Known limitations

- **Model recall on cancer is 0.68** at the default threshold — tune the threshold on a
  labelled validation set before any real use.
- `db.create_all()` creates missing tables but does not migrate schema changes; introduce
  Flask-Migrate/Alembic before evolving the models.
- No rate limiting on `/analyzer`, and one image per request; no batch or async scoring.
- The dataset is a small binary image set with no patient-level train/val split, so
  metrics can be optimistic relative to a true clinical hold-out.
- The estimator consumes mammogram-style images only; photos of other kinds will return
  meaningless probabilities.
