# Contributing

1. Fork the repository and create a focused branch.
2. Create a Python 3.11+ virtual environment.
3. Run `python -m pip install -e '.[dev]'`.
4. Run `make lint` and `make test` before opening a pull request.

Explain the behavior change, add tests, and keep data provenance explicit. Do not
commit API keys, raw provider responses, model binaries, or private data. Report model
results from chronological race splits, never random row splits. By contributing, you
agree that your contribution is licensed under the MIT License.

