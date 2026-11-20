# NASCAR Performance Lab

[![CI](https://github.com/ArjjunRay/nascar-performance-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/ArjjunRay/nascar-performance-lab/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

NASCAR Performance Lab is an open-source pipeline and API for experimenting with
historical driver-performance models. It collects race data, builds features without
using future results, evaluates models on later races, and serves relative finish
rankings through FastAPI.

> [!IMPORTANT]
> This is research software, not a betting system. The bundled workflow generates
> synthetic data only to exercise the software. Demo metrics are not evidence of
> real-world predictive performance.

## What is included

- A retrying SportsDataIO collection client with environment-based secrets
- Reproducible raw-to-tabular data preparation
- Leakage-safe rolling and track-history features
- Race-level chronological train/test splitting
- Ridge and random-forest baselines with automatic selection by holdout MAE
- Relative ranking and pairwise ordering metrics
- A versioned FastAPI service with validation, health checks, and OpenAPI docs
- A command-line workflow, Docker image, Render blueprint, tests, and CI

## Architecture

```text
SportsDataIO API
      |
      v
data/raw/*.json  -- prepare -->  data/processed/*.csv
                                      |
                                      v
                            historical features
                                      |
                                      v
                         chronological evaluation
                                      |
                                      v
                    models/nascar_rank_model.joblib
                                      |
                                      v
                          FastAPI /v1/predictions
```

## Quick start

Requirements: Python 3.11 or newer.

```bash
git clone https://github.com/ArjjunRay/nascar-performance-lab.git
cd nascar-performance-lab
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Generate synthetic data and train a demonstration artifact:

```bash
nascar-performance demo-data --output-dir data/demo
nascar-performance train --data-dir data/demo
```

Start the API:

```bash
uvicorn nascar_performance.api:app --reload
```

Open <http://127.0.0.1:8000/docs> for interactive documentation. A complete request
is available in [`docs/api-example.json`](docs/api-example.json).

## Use fresh data

Create an API key through SportsDataIO, then set it only in your environment. Never
commit it.

```bash
export SPORTSDATAIO_KEY='your-key'
nascar-performance collect --season 2025
nascar-performance prepare
nascar-performance train
```

The collector writes provider responses under `data/raw/`, which Git ignores. Before
training, confirm that your subscription returns non-scrambled position fields. Do not
pass `--normalize-scrambled` for real position data.

## API example

```bash
curl -X POST http://127.0.0.1:8000/v1/predictions/rank \
  -H 'Content-Type: application/json' \
  --data @docs/api-example.json
```

`projected_finish_position` is the ordering among the submitted entrants.
`model_score` is the model's raw finish-position estimate and is not a probability.

## Docker and deployment

Build and run locally. The image build creates its demonstration model from the bundled
sample:

```bash
docker compose up --build
```

The image runs as a non-root user and exposes `/healthz`. `render.yaml` provides a
Render blueprint; any container host can deploy the same image. See
[`docs/deployment.md`](docs/deployment.md) for production configuration and limitations.

## Development

```bash
make lint
make test
```

The exploratory notebook calls the tested package instead of maintaining a second
implementation: [`notebooks/model_evaluation.ipynb`](notebooks/model_evaluation.ipynb).

## Data and model limitations

- Generated demo data is synthetic and cannot support real-world performance claims.
- A driver's current metadata is not assumed to be historical race-time metadata.
- The model omits weather, practice, qualifying changes, mechanical failures, pit
  strategy, and roster changes.
- The final artifact is fitted on all available rows after model selection; JSON
  metadata preserves results from the chronological holdout.

Read [`data/README.md`](data/README.md) before adding or replacing data.

## Contributing and security

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`SECURITY.md`](SECURITY.md) before
opening a pull request or reporting a vulnerability.

## Contributors

Arjun Ray, Aaryaman Bisht, Adhiraj Nimbalkar, Tolu Fashina, and Noah Zemba.

## License

Code is available under the [MIT License](LICENSE). Data remains subject to its
provider's terms and is not relicensed by this repository.
