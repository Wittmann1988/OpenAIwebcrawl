#!/bin/bash
set -euo pipefail

# Only run in remote (Claude Code on the web) environments
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

# Install Python dependencies
# Filter out packages that fail to build on modern Python / aren't needed at runtime:
#   - bs4==0.0.1: wrapper; beautifulsoup4 (in requirements) provides the 'bs4' module
#   - docopt==0.6.2: fails to build wheel on Python 3.11+
#   - pipreqs==0.4.12: dev tool (depends on docopt), not needed at runtime
# --ignore-installed: avoids conflicts with system-managed packages (e.g. PyYAML)
grep -vE '^(bs4|docopt|pipreqs)==' "$CLAUDE_PROJECT_DIR/requirements.txt" \
  | pip install --ignore-installed -r /dev/stdin

# Install flake8 for linting
pip install flake8
