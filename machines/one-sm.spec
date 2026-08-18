# One SM, filled. Isolates occupancy from parallelism: the same machine as
# default.spec with room for every block the warp slots allow, so what changes
# against it is blocks covering each other's waiting and nothing else.
sm_count            = 1
blocks_per_sm       = 32
warp_slots_per_sm   = 64
shared_bytes_per_sm = 16384
l1_lines            = 1024
l2_lines            = 65536
