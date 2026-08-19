#!/usr/bin/env bash
# Apply .clang-format to every C++ source in the tree.
#
#   scripts/format.sh          rewrite files in place
#   scripts/format.sh --check  report unformatted files, exit 1 if any (for CI)
#
# Equivalent CMake targets: `cmake --build build --target format` / `format-check`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Not every macOS install exposes clang-format on PATH; the Xcode Command Line
# Tools ship one that xcrun can locate.
find_clang_format() {
    if command -v clang-format >/dev/null 2>&1; then
        command -v clang-format
    elif command -v xcrun >/dev/null 2>&1 && xcrun -f clang-format >/dev/null 2>&1; then
        xcrun -f clang-format
    else
        echo "error: clang-format not found." >&2
        echo "  macOS: xcode-select --install   (or: brew install clang-format)" >&2
        echo "  Linux: apt install clang-format" >&2
        return 1
    fi
}

CLANG_FORMAT="$(find_clang_format)"

# Only our own sources — build/ holds fetched dependencies.
# Skip a directory that is not there, so a partial checkout still formats.
SEARCH_DIRS=()
for d in gpurt apps test; do
    [[ -d "$ROOT/$d" ]] && SEARCH_DIRS+=("$ROOT/$d")
done

# No mapfile here: macOS ships bash 3.2, which predates it.
FILES=()
if [[ ${#SEARCH_DIRS[@]} -gt 0 ]]; then
    while IFS= read -r f; do
        FILES+=("$f")
    done < <(find "${SEARCH_DIRS[@]}" -type f \
        \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' \) | sort)
fi

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "no sources found"
    exit 0
fi

if [[ "${1:-}" == "--check" ]]; then
    unformatted=()
    for f in "${FILES[@]}"; do
        if ! diff -q "$f" <("$CLANG_FORMAT" "$f") >/dev/null 2>&1; then
            unformatted+=("${f#"$ROOT"/}")
        fi
    done

    if [[ ${#unformatted[@]} -gt 0 ]]; then
        echo "not formatted (${#unformatted[@]}):"
        printf '  %s\n' "${unformatted[@]}"
        echo
        echo "run: scripts/format.sh"
        exit 1
    fi
    echo "all ${#FILES[@]} files formatted"
    exit 0
fi

"$CLANG_FORMAT" -i "${FILES[@]}"
echo "formatted ${#FILES[@]} files"
