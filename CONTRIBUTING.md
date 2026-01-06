# Contributing to Good AI Enterprise Framework

Thank you for your interest in contributing!

## Quick Start

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/enterprise-ai-framework.git
cd enterprise-ai-framework

# Setup development environment
make setup

# Run tests to verify setup
make test
```

## Development Workflow

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards

3. **Run checks**:
   ```bash
   make lint   # Check code style
   make test   # Run tests
   ```

4. **Commit** with a clear message:
   ```bash
   git commit -m "feat: add new feature description"
   ```

5. **Push and create a Pull Request**

## Coding Standards

### Python Style

- Follow PEP 8
- Use type hints
- Maximum line length: 100 characters
- Use `ruff` for linting (included in dev dependencies)

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation only
- `test:` - Adding/updating tests
- `refactor:` - Code change that neither fixes a bug nor adds a feature
- `chore:` - Maintenance tasks

### Testing Requirements

- All new features must include tests
- Maintain or improve test coverage (currently 73%)
- Tests must be deterministic (no random failures)
- Use pytest fixtures for shared setup

### Documentation

- Update README.md if adding user-facing features
- Add docstrings to public functions
- Update CHANGELOG.md for notable changes

## Pull Request Process

1. Ensure all tests pass
2. Update documentation as needed
3. Add entry to CHANGELOG.md under "Unreleased"
4. Request review from maintainers
5. Address review feedback
6. Maintainer will merge when approved

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Be respectful and constructive.

## Questions?

- Open an issue for bugs or feature requests
- Tag with appropriate labels

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
