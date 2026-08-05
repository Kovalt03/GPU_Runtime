#!/usr/bin/env bash
# Configure and build.
#
#   scripts/build.sh                 build (configuring first if needed)
#   scripts/build.sh --clean         discard build/ and start over
#   scripts/build.sh --release       build with optimisations
#
# Configuring is skipped when build/ is already usable, so repeat builds are
# incremental. A build/ left behind by a different generator — switching between
# Make and Ninja, most often by installing one of them — makes CMake refuse to
# configure while ctest happily keeps running the stale binaries from the last
# successful build. That failure is silent and looks like passing tests, so it
# is detected here and the directory is regenerated.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/build"
BUILD_TYPE="Debug"

for arg in "$@"; do
    case "$arg" in
    --clean) rm -rf "$BUILD_DIR" ;;
    --release) BUILD_TYPE="Release" ;;
    -h | --help)
        sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        exit 0
        ;;
    *)
        echo "error: unknown option $arg" >&2
        exit 1
        ;;
    esac
done

configure() {
    cmake -S "$ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
}

if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    configure
elif ! configure; then
    echo
    echo "configure failed; regenerating $BUILD_DIR from scratch"
    rm -rf "$BUILD_DIR"
    configure
fi

# --parallel with no number lets CMake pick, which avoids hardcoding nproc —
# absent on macOS anyway.
cmake --build "$BUILD_DIR" --parallel
