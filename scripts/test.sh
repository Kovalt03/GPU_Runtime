#!/usr/bin/env bash
# Build, then run the test suite.
#
#   scripts/test.sh                  everything
#   scripts/test.sh Scheduler        only tests whose name matches
#   scripts/test.sh Isa.Vcmp         a single case
#
# Building first is the point: ctest runs whatever binaries are already in
# build/, so calling it directly after an edit can report on code that was never
# compiled.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"$ROOT/scripts/build.sh"

echo
if [[ $# -gt 0 ]]; then
    ctest --test-dir "$ROOT/build" --output-on-failure -R "$1"
else
    ctest --test-dir "$ROOT/build" --output-on-failure
fi
