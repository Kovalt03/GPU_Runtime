# Four SMs holding two blocks each — small enough that a launch can fill it and
# large enough that two can share it.
#
# The default machine is one SM holding one block, where nothing ever overlaps:
# a queue drained mid-frame loses nothing, because there was nothing running
# beside it. v100 is the other extreme, 80 SMs that swallow anything these
# benchmarks launch. This one is where an overlap is worth measuring.
sm_count            = 4
blocks_per_sm       = 2
