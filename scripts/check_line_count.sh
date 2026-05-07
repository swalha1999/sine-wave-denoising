#!/usr/bin/env bash
# Enforce the submission guideline that every Python file under src/ and tests/
# contains at most 150 lines of code (blank lines and comment-only lines do not
# count). Exits 0 when all files comply, 1 otherwise.

set -euo pipefail

LIMIT="${LINE_COUNT_LIMIT:-150}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

violations=0
while IFS= read -r -d '' f; do
    loc=$(awk '
        {
            line = $0
            sub(/^[[:space:]]+/, "", line)
            if (line == "") next
            if (substr(line, 1, 1) == "#") next
            count++
        }
        END { print count + 0 }
    ' "$f")
    if [ "$loc" -gt "$LIMIT" ]; then
        printf '%s: %d lines of code (limit %d)\n' "$f" "$loc" "$LIMIT" >&2
        violations=$((violations + 1))
    fi
done < <(find src tests -type f -name '*.py' -print0)

if [ "$violations" -gt 0 ]; then
    printf '\n%d file(s) exceed the %d-line limit.\n' "$violations" "$LIMIT" >&2
    exit 1
fi

printf 'All Python files under src/ and tests/ are within the %d-line limit.\n' "$LIMIT"
