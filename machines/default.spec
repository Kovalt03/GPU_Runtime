# The machine every figure in benchmarks/RESULTS.md was taken on.
#
# One SM holding one block is what myrt_launch used to be — a nested loop over
# the grid — so this file describes the simulator's history rather than any
# hardware. Every other file here is read against it.
sm_count            = 1
blocks_per_sm       = 1
warp_slots_per_sm   = 64
shared_bytes_per_sm = 16384
l1_lines            = 1024
l2_lines            = 65536
