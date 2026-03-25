# Contributing Guide

Thanks for your interest in improving this HCC research project.

## Development Workflow

1. Create a feature branch.
2. Keep pull requests focused and small.
3. Add/update documentation for any user-visible change.
4. Run local checks before opening a PR.

## Local Validation

- Ensure Python dependencies are installed from `requirements.txt`.
- Verify key scripts run without syntax errors.
- Keep notebooks reproducible (clear noisy outputs when possible).

## Commit Style

Use clear commit messages, e.g.:

- `feat: add full-volume inference summary`
- `fix: correct HU windowing edge case`
- `docs: update reproducibility section`

## Pull Request Checklist

- [ ] Code compiles/runs locally
- [ ] Documentation updated
- [ ] No sensitive data or large binaries accidentally included
- [ ] Medical disclaimer preserved where relevant
