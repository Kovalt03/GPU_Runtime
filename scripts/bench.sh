#!/usr/bin/env bash
# Every renderer and benchmark, into one directory named for when it ran.
#
#   scripts/bench.sh          benchmarks/result/<timestamp>/, at 256
#   scripts/bench.sh 512      the same, four times the pixels and four times
#                             the wall clock — the walk route is what costs
#
# Runs accumulate: nothing here deletes an earlier one. The directory is
# gitignored apart from .gitkeep, so the record is local — the figures that
# belong to the repository go in RESULTS.md beside it, pasted from
# render_bench.md.
#
# The seconds each program reports are wall clock on this host and do not
# reproduce. The lane and warp counts beside them do, and are what to compare.
#
# Built with optimisations for the same reason — see below.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SIZE="${1:-256}"

# Release, because these programs report seconds. Every instruction they time
# runs inside libgpurt, so a Debug build measures the debug build — the mesh
# sweep takes 41 seconds there against 7 here, for the same issued work.
"$ROOT/scripts/build.sh" --release

RUN="benchmarks/result/$(date +%Y-%m-%d_%H%M%S)"
mkdir -p "$RUN"

step() {
    echo
    echo "=== $1 ==="
}

step "ray_triangle"
./build/kernels/ray_triangle "$SIZE" "$SIZE" --out "$RUN"

step "raster_triangle"
./build/kernels/raster_triangle "$SIZE" "$SIZE" 1 --out "$RUN"

# The two above render the same triangle by different arithmetic, so their files
# have to match byte for byte. It is the check neither can make alone.
if cmp -s "$RUN/images/result.ppm" "$RUN/images/raster.ppm"; then
    echo "ray_triangle and raster_triangle agree byte for byte"
else
    echo "MISMATCH between images/result.ppm and images/raster.ppm" >&2
    exit 1
fi

step "mesh_render"
./build/kernels/mesh_render --size "$SIZE" --out "$RUN"

step "render_bench"
./build/benchmarks/render_bench --out "$RUN"

step "divergence_bench"
./build/benchmarks/divergence_bench | tee "$RUN/divergence.txt"

{
    echo "# Run $(date '+%Y-%m-%d %H:%M:%S')"
    echo
    echo "- meshes: everything in assets/"
    echo "- resolution: ${SIZE}x${SIZE}"
    echo "- commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'not a repository')"
    echo "- host: $(uname -sm)"
    echo "- build: $(grep CMAKE_BUILD_TYPE: build/CMakeCache.txt | cut -d= -f2)"
    echo
    echo "Seconds in these files are wall clock on this host and on that build"
    echo "type — the same sweep takes six times as long from a Debug one, every"
    echo "instruction being dispatched inside libgpurt. The lane and warp counts"
    echo "reproduce anywhere."
} > "$RUN/run.md"

echo
echo "wrote $RUN"
ls -1 "$RUN"
echo "  images/ holds $(ls -1 "$RUN/images" | wc -l | tr -d " ") PPMs"
