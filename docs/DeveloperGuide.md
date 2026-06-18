# Developer & Contribution Guide

Welcome to the Project AEGIS developer portal. Use this guide to set up your local development environment and run testing suites.

## Local Setup

### Prerequisites
- Python 3.10 or 3.11
- Poetry (version 1.5.0+)

### Step-by-Step Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/project-aegis.git
   cd project-aegis
   ```
2. Install dependencies via Poetry:
   ```bash
   poetry install
   ```
3. Set up pre-commit hooks:
   ```bash
   poetry run pre-commit install
   ```

## Development Standards

### Code Formatting and Linting
We use **Ruff** for fast linting and formatting. Run:
```bash
poetry run ruff check .
poetry run ruff format .
```

### Static Type Checks
Run **Mypy** to check types:
```bash
poetry run mypy --ignore-missing-imports src/ packages/
```

### Security Audit
Run **Bandit** to scan for vulnerabilities:
```bash
poetry run bandit -r src/ packages/
```

## Testing Strategy
We use **Pytest** for testing. Run:
```bash
poetry run pytest tests/
```
Ensure code coverage does not drop below 80% on new pull requests.
