# Deployment

The container build generates synthetic data and trains a demonstration artifact so a
fresh checkout is deployable without committing data or an opaque model binary:

```bash
docker build -t nascar-performance-lab .
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_PATH` | `models/nascar_rank_model.joblib` | Model artifact path |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `PORT` | `8000` | Container listen port |

The inference service does not need `SPORTSDATAIO_KEY`; run collection as a separate,
controlled job. `render.yaml` is a deployment blueprint, not evidence that a live
service exists. After deployment, confirm `/healthz` reports `model_loaded: true` and
test `/v1/predictions/rank` with `docs/api-example.json`.

Before serious public use, store artifacts in a registry, add authentication and rate
limiting, publish service-level objectives, emit structured metrics, monitor drift, and
retrain only on licensed non-scrambled data.

