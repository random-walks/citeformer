# Authors

citeformer was authored by:

- **[Blaise Albis-Burdige](https://blaiseab.com)** ([@blaiseab](https://github.com/blaiseab)) — design, implementation, benchmarks, docs.

Contributors welcome — open a PR and add your name here. See [CONTRIBUTING.md](CONTRIBUTING.md) for the dev loop.

## Acknowledgements

The core insight — that citation fabrication is a *structural* problem solvable by constraining the model's output grammar, not a behavioural one solvable by prompting — is borrowed from [jsonformer](https://github.com/1rgs/jsonformer) by [Nick Kapur](https://github.com/1rgs). jsonformer applied it to JSON schemas in 2023; citeformer applies it to citation markers in 2026.

The heavy lifting lives in dependencies maintained by others — see the [piggyback map](docs/reference/architecture.md#piggyback-first). The library's contribution is composition, not reinvention.
