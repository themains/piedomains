# Trusted Publishing Configuration Fix

## Problem Analysis

The error shows:
- Repository: `themains/know-your-ip`
- Environment: `MISSING` ← **This is the key issue**
- Branch: `master`
- Workflow: `python-publish.yml`

## Required Actions

### 1. Update Workflow File in `themains/know-your-ip`

Replace the `python-publish.yml` file in `themains/know-your-ip` with this exact content:

```yaml
# Publish to PyPI when a new release is created or manually triggered
# Uses OpenID Connect trusted publishing for enhanced security
name: python-publish

on:
  release:
    types: [published]
  workflow_dispatch:  # Enables manual triggering

permissions:
  contents: read
  id-token: write  # Required for trusted publishing with OIDC

jobs:
  publish:
    runs-on: ubuntu-latest
    environment: pypi  # ← THIS IS CRITICAL - currently missing!
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install uv
        uses: astral-sh/setup-uv@v3
        with:
          version: "latest"
      
      - name: Build package
        run: uv build
      
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        # Trusted publishing - no API token needed
```

### 2. Configure Trusted Publisher on PyPI

Go to your PyPI project (know-your-ip) settings:

1. **Visit**: https://pypi.org/manage/project/know-your-ip/settings/publishing/
2. **Add trusted publisher** with these **EXACT** values:

   - **Repository owner**: `themains`
   - **Repository name**: `know-your-ip`
   - **Workflow filename**: `python-publish.yml`
   - **Environment name**: `pypi`

### 3. Create PyPI Environment in GitHub

In the `themains/know-your-ip` repository:

1. Go to **Settings** → **Environments**
2. Click **New environment**
3. Name it: `pypi`
4. (Optional) Add protection rules like requiring reviews

### 4. Test the Configuration

1. Create a test release in `themains/know-your-ip`
2. Check the workflow logs for:
   ```
   * `environment`: `pypi`  # Should no longer be MISSING
   ```
3. Verify successful publication

## Verification Checklist

- [ ] Workflow file updated with `environment: pypi`
- [ ] PyPI trusted publisher configured for `themains/know-your-ip`
- [ ] GitHub environment `pypi` created
- [ ] Test release triggers workflow successfully
- [ ] Environment claim appears in logs (not MISSING)

## Security Benefits

✅ **No API tokens to manage**  
✅ **Automatic token rotation**  
✅ **Audit trail through OIDC**  
✅ **Reduced credential exposure**

## Troubleshooting

If you still get errors:

1. **Double-check repository names** (themains/know-your-ip)
2. **Verify branch name** (master vs main)
3. **Confirm PyPI project name** matches exactly
4. **Check environment name** is exactly `pypi`

The key fix is adding `environment: pypi` to the workflow, which will make the environment claim appear in the OIDC token instead of being `MISSING`.