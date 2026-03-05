# Contributing to AskMyDocs

Thank you for taking the time to contribute! This guide covers everything you need to submit a high-quality pull request.

---

## Table of contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting started](#getting-started)
3. [Development workflow](#development-workflow)
4. [Commit messages](#commit-messages)
5. [Pull request checklist](#pull-request-checklist)
6. [Issue reporting](#issue-reporting)

---

## Code of Conduct

This project follows the [Contributor Covenant v2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
Be respectful, constructive, and welcoming.

---

## Getting started

```bash
git clone https://github.com/Muhammad-Farooq-13/ask-my-docs.git
cd ask-my-docs
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
make dev-install
cp .env.example .env
```

Run the test suite before making any changes to confirm your baseline is green:

```bash
make test
```

---

## Development workflow

1. **Fork** the repository and create a feature branch:
   ```bash
   git switch -c feat/short-description
   ```
2. Make focused, incremental changes — one logical change per commit.
3. Add or update tests for every behaviour change. Coverage must not decrease.
4. Run the full quality suite before pushing:
   ```bash
   make lint test
   ```
5. Open a Pull Request against `main` using the PR template.

---

## Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short imperative summary>

[optional body explaining WHY, not WHAT]

[optional footer: Closes #123, BREAKING CHANGE: ...]
```

| Type | When to use |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `perf` | Performance improvement |
| `refactor` | Code change that is neither fix nor feat |
| `test` | Adding or fixing tests |
| `docs` | Documentation only |
| `ci` | CI/CD pipeline changes |
| `chore` | Maintenance (deps, config) |

---

## Pull request checklist

Before submitting, confirm every item below:

- [ ] `make lint` — zero errors, zero warnings
- [ ] `make test` — all tests pass
- [ ] New code has corresponding unit tests
- [ ] No secrets, API keys, or credentials committed
- [ ] `CHANGELOG.md` entry added under `[Unreleased]`
- [ ] PR title follows Conventional Commits

---

## Issue reporting

Please use the GitHub Issue templates:

- **Bug reports** → choose "Bug report"
- **Feature requests** → choose "Feature request"

For security vulnerabilities, see [SECURITY.md](SECURITY.md). **Do not** open a public issue for security bugs.
