# Contributing to nano-t2i

Thank you for your interest in contributing. This project is meant to stay small and hackable; thoughtful fixes, clearer docs, and well-scoped improvements are especially welcome.

## Ways to contribute

You can help in several ways:

- **Bug reports** — Open a [GitHub issue](https://github.com/gojasper/nano-t2i/issues) with a minimal reproduction, your environment (Python version, CUDA driver, GPU), and relevant logs or stack traces.
- **Feature requests** — Describe the use case and why it fits the project’s goals (minimal, reproducible T2I training on MONET). Large architectural changes are easier to discuss in an issue before you invest in a PR.
- **Code changes** — Bug fixes, training/data pipeline improvements, model or config extensions, and demo updates in `src/nano_t2i/` or `examples/`.
- **Documentation** — README clarifications, config comments, or examples that help others reproduce training or inference.
- **Configs** — New or improved YAML configs under `examples/trainings/configs/` (keep them documented and runnable where possible).

Please do not open PRs that only reformatted unrelated files or bump dependencies without discussion unless you are fixing a concrete breakage.

## Development setup

1. **Fork and clone** the repository:

   ```shell
   git clone https://github.com/<your-username>/nano-t2i.git
   cd nano-t2i
   ```

2. **Create a Python 3.13 environment** and install the training extras (see [README.md](README.md#setup)):

   ```shell
   uv venv envs/nano-t2i --python 3.13
   source envs/nano-t2i/bin/activate
   uv pip install -e ".[training]"
   ```

   You need CUDA 12.8–compatible drivers for the pinned PyTorch wheels in `requirements.txt`.


Full dataset and training instructions are in [README.md](README.md).

## Project layout

When changing code, it helps to know where things live:

| Path | Purpose |
|------|---------|
| `src/nano_t2i/` | Core library: models, data pipeline, trainer, configs |
| `examples/trainings/` | Training entrypoint and YAML configs |
| `examples/inference/demo/` | Gradio demo |
| `examples/trainings/configs/` | Reference and experimental training configs |

Architectural and training hyperparameters are usually driven by YAML configs (e.g. [`examples/trainings/configs/nano.yaml`](examples/trainings/configs/nano.yaml)) rather than hard-coded constants. Prefer extending configs when your change is meant to be user-tunable.

## Code style

Match the existing codebase:

- **Python** ≥ 3.13; use type hints where the surrounding module already does.
- **Formatting** — Run [Black](https://github.com/psf/black) and [isort](https://pycqa.github.io/isort/) before submitting (both are listed in `requirements-training.txt`):

  ```shell
  black src examples
  isort src examples
  ```

- **Naming** — Follow local conventions (e.g. `*_config.py` for Pydantic/dataclass configs, `snake_case` for functions and modules).
- **Imports** — Keep third-party and local imports grouped consistently with nearby files.
- **Adapted code** — If you port or adapt external code (as in the FLUX-derived transformer modules), keep the source attribution comment at the top of the file.

There is no separate test suite in this repository yet. For behavioral changes, describe in the PR how you validated the change (e.g. short training run, demo smoke test, config load).

## Making a pull request

1. **Discuss large changes** — Open an issue first for broad refactors, new dependencies, or API-breaking changes.
2. **Branch** — Work on a topic branch from `main` (e.g. `fix/dataloader-shard-key`, `docs/training-wandb`).
3. **Scope** — Keep PRs focused: one logical change per PR is easier to review.
4. **Commits** — Use clear, imperative messages (e.g. `Fix collation when batch size is 1`, `Document phase-2 checkpoint resume`).
5. **Open the PR** against `gojasper/nano-t2i` `main` and fill in:
   - **What** changed
   - **Why** it is needed
   - **How** you tested it (commands, hardware if relevant)
   - Any **breaking** or config changes

Maintainers may ask for edits or suggest splitting large PRs; that is normal.

## Reporting security issues

Please do not report security vulnerabilities in public issues. Contact the maintainers listed in [`pyproject.toml`](pyproject.toml) privately so we can coordinate a fix.

## License

By contributing, you agree that your contributions will be licensed under the same [Apache License 2.0](LICENSE) as the rest of the project. You must have the right to submit the work you contribute (your own code, or code under a compatible license with proper attribution).

## Questions

- **Usage / setup** — Check [README.md](README.md) and existing [issues](https://github.com/gojasper/nano-t2i/issues).
- **MONET dataset** — See the [dataset card](https://huggingface.co/datasets/jasperai/monet).

We appreciate every contribution that keeps nano-t2i clear, reproducible, and easy to hack on.
