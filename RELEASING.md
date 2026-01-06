# Releasing

This document describes the release process for Good AI Enterprise Framework.

## Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** (x.0.0): Breaking changes
- **MINOR** (0.x.0): New features, backward compatible
- **PATCH** (0.0.x): Bug fixes, backward compatible

## Release Checklist

### Before Release

1. **Ensure CI passes** on the main branch
   ```bash
   make lint
   make test
   ```

2. **Update version** in `pyproject.toml`:
   ```toml
   version = "X.Y.Z"
   ```

3. **Update CHANGELOG.md**:
   - Move items from `[Unreleased]` to new version section
   - Add release date
   - Update comparison links at bottom

4. **Create release commit**:
   ```bash
   git add pyproject.toml CHANGELOG.md
   git commit -m "chore: release vX.Y.Z"
   ```

5. **Create and push tag**:
   ```bash
   git tag -a vX.Y.Z -m "Release vX.Y.Z"
   git push origin main --tags
   ```

### After Tagging

6. **Create GitHub Release**:
   - Go to Releases > Draft new release
   - Select the tag
   - Title: `vX.Y.Z`
   - Copy relevant CHANGELOG section to description
   - Publish release

7. **Verify release**:
   - Check GitHub Actions completed
   - Verify release appears in releases list

## Hotfix Process

For critical fixes to released versions:

1. Create branch from tag:
   ```bash
   git checkout -b hotfix/vX.Y.Z vX.Y.Z
   ```

2. Apply fix and update version to `X.Y.(Z+1)`

3. Follow standard release process

## Pre-release Versions

For alpha/beta releases:

```
0.2.0-alpha.1
0.2.0-beta.1
0.2.0-rc.1
```

Tag with full version: `v0.2.0-alpha.1`
