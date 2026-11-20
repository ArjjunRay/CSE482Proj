# Data

Run `nascar-performance demo-data --output-dir data/demo` to create a deterministic
synthetic dataset. It exists only for tests, examples, and deployment smoke checks; its
metrics do not describe real NASCAR performance.

SportsDataIO may mark numeric fields as scrambled depending on the subscription. The
optional `--normalize-scrambled` training flag converts within-race ordering to ordinal
ranks, but it cannot reconstruct original values. Review provider terms before storing,
redistributing, or using collected data.

- `raw/`: local API responses; ignored by Git
- `processed/`: locally prepared CSVs
- `demo/`: generated synthetic tables; ignored by Git

Never commit credentials, personal data, or data you cannot redistribute.

