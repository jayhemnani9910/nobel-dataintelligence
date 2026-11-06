# Contributing

Thanks for looking. This is a research codebase, so small fixes, tests and honest benchmark runs help most.

## Setup

Python 3.10 to 3.12.

```bash
git clone https://github.com/jayhemnani9910/nobel-dataintelligence.git
cd nobel-dataintelligence
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt ruff
```

Run everything from the repo root. The code imports `src` and `vibropredict` from there.

## Tests and lint

```bash
python -m pytest tests/ vibropredict/tests/ --ignore=tests/smoke/
python -m pytest tests/smoke/          # short training run, a minute or so on CPU
ruff check .
ruff format --check .
```

CI runs the same commands on Python 3.10, 3.11 and 3.12. Tests must not download models or datasets. Mock them or skip when they are missing.

## Style

- `ruff format` decides formatting. Settings are in `ruff.toml`.
- Keep changes small and focused. Add a test for any bug you fix.
- Do not report a result in docs or the README unless a script in the repo produces it.

## Branches and pull requests

- Branch from `master`, open the PR against `master`.
- Commit messages follow the style already in the log: `fix(physics): ...`, `feat(data): ...`, `docs: ...`.
- Say in the PR what you changed and how you tested it.

## Bug reports

Open an issue with the steps to reproduce, what you expected, what happened, your OS, Python version and the full traceback. If it involves a protein, give the PDB ID.

I try to reply to issues within a few days.
