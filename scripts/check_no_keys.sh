#!/usr/bin/env bash
# Fails if a service-account key is tracked in git, matched by file name and by
# content (so a renamed key is caught too). Run from the repo root.
# shortcut: scans tracked files only; a key sitting untracked is stopped by
# .gitignore, not here.
set -u
self=scripts/check_no_keys.sh
test=backend/tests/test_credential_guard.py  # holds the patterns on purpose
bad=0

names=$(git ls-files | grep -iE '(service-?account|firebase-adminsdk)[^/]*\.json$')
if [ -n "$names" ]; then echo "Key-like file name:"; echo "$names"; bad=1; fi

content=$(git ls-files -z | grep -zvxF -e "$self" -e "$test" \
  | xargs -0 grep -IlE '"type"[[:space:]]*:[[:space:]]*"service_account"|"private_key"[[:space:]]*:' 2>/dev/null)
if [ -n "$content" ]; then echo "Key-like content:"; echo "$content"; bad=1; fi

exit $bad
