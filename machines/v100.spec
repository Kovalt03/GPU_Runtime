# Volta, 2017 — an approximation, not a spec sheet.
#
# The SM count and the per-SM limits are the published ones. What is *not*
# modelled is everything below the block: four warp schedulers an SM, dual issue,
# and a memory system SMs contend for. This machine issues one instruction an SM
# a cycle and its SMs never queue behind each other, so a figure taken here is a
# bound rather than a prediction.
sm_count            = 80
blocks_per_sm       = 32
warp_slots_per_sm   = 64
shared_bytes_per_sm = 98304
l1_lines            = 1024
l2_lines            = 49152
