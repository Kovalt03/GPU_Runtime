#!/usr/bin/env bash
# Every gate at once: formatting, a warning-free build, and the test suite.
# Run this before committing, and from CI.
#
# Warnings are errors here but not in an ordinary build, so that work in
# progress stays comfortable while what gets committed does not accumulate them.
#
# That flag is also why this configures a tree of its own rather than reusing
# the working one: sharing it would flip -Werror on for every ordinary build
# too. It sits under build/ so there is still only one directory to ignore, and
# being separate is what makes it the clean tree a commit is checked against.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/build/check"

failed=0
step() {
    echo
    echo "=== $1 ==="
}

step "format"
if "$ROOT/scripts/format.sh" --check; then
    echo "ok"
else
    failed=1
fi

step "build (warnings as errors)"
if cmake -S "$ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS="-Werror" >/dev/null &&
    cmake --build "$BUILD_DIR" --parallel; then
    echo "ok"
else
    failed=1
fi

step "tests"
if ctest --test-dir "$BUILD_DIR" --output-on-failure; then
    echo "ok"
else
    failed=1
fi

echo
if [[ $failed -ne 0 ]]; then
    echo "FAILED"
    exit 1
fi
echo "all checks passed"
