#include <algorithm>
#include <cstdio>

#include "gpu_spec.hpp"

uint32_t GPUSpec::residency(uint32_t warps_per_block, size_t shared_bytes) const
{
    // The smallest of the three, and never zero: a block that fits none of the
    // limits still has to run, or a kernel asking for too much would report a
    // machine with no room rather than an error anyone could act on.
    uint32_t by_warps = sms.warp_slots_per_sm;
    if (warps_per_block > 0) {
        by_warps = sms.warp_slots_per_sm / warps_per_block;
    }
    uint32_t by_shared = sms.blocks_per_sm;
    if (shared_bytes > 0) {
        by_shared = static_cast<uint32_t>(sms.shared_bytes_per_sm / shared_bytes);
    }

    const uint32_t allowed = std::min(sms.blocks_per_sm, std::min(by_warps, by_shared));
    return allowed == 0 ? 1u : allowed;
}

std::string GPUSpec::describe() const
{
    // Sizes as they are configured, and the fixed ones beside them: a reader
    // asking what machine a table came from wants both, and only one of the two
    // can be changed.
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "  SMs                %u, %u block(s) each\n"
                  "  warp slots an SM   %u  (%u lanes a warp)\n"
                  "  shared an SM       %zu bytes  (%u floats a block)\n"
                  "  registers a thread %u\n"
                  "  L1 / L2            %zu / %zu lines of %u bytes\n"
                  "  line cost          L1 %u, L2 %u, miss by opcode\n"
                  "  line latency       L1 %u, L2 %u, memory %u\n",
                  sms.sm_count, sms.blocks_per_sm, sms.warp_slots_per_sm, WARP_SIZE,
                  sms.shared_bytes_per_sm, SHARED_MEM_FLOATS, REGS_PER_THREAD, l1_lines,
                  l2_lines, CACHE_LINE_BYTES, L1_HIT_COST, L2_HIT_COST, L1_HIT_LATENCY,
                  L2_HIT_LATENCY, MEMORY_LATENCY);
    return buf;
}
