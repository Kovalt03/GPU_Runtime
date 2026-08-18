#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

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

std::string GPUSpec::to_text() const
{
    // The fields a machine file may set, in the order describe() prints them, so
    // that a written file and a printed header read alike.
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "sm_count            = %u\n"
                  "blocks_per_sm       = %u\n"
                  "warp_slots_per_sm   = %u\n"
                  "shared_bytes_per_sm = %zu\n"
                  "l1_lines            = %zu\n"
                  "l2_lines            = %zu\n",
                  sms.sm_count, sms.blocks_per_sm, sms.warp_slots_per_sm,
                  sms.shared_bytes_per_sm, l1_lines, l2_lines);
    return buf;
}

namespace {

// Names a machine file cannot set, and why. Rejected by name rather than ignored
// so that a file asking for a 64-lane warp is told that this is not a knob
// instead of quietly running with 32.
const char* fixed_field(const std::string& name)
{
    if (name == "warp_size") {
        return "WARP_SIZE sizes std::array<Thread, 32> in thread.hpp";
    }
    if (name == "regs_per_thread") {
        return "REGS_PER_THREAD sizes std::array<float, 256> in thread.hpp";
    }
    if (name == "shared_mem_floats" || name == "shared_bytes_per_block") {
        return "SHARED_MEM_FLOATS sizes ThreadBlock's scratchpad; a launch says "
               "how much of it it uses, through LaunchConfig::shared_bytes";
    }
    if (name == "cache_line_bytes") {
        return "CACHE_LINE_BYTES is what every address is divided by, and the "
               "figures in benchmarks/ are taken at 128";
    }
    return nullptr;
}

std::string trimmed(const std::string& text)
{
    const size_t first = text.find_first_not_of(" \t\r");
    if (first == std::string::npos) {
        return {};
    }
    return text.substr(first, text.find_last_not_of(" \t\r") - first + 1);
}

uint64_t to_number(const std::string& name, const std::string& value)
{
    try {
        size_t consumed = 0;
        const unsigned long long parsed = std::stoull(value, &consumed);
        if (consumed != value.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return parsed;
    } catch (const std::exception&) {
        throw std::runtime_error("machine spec: " + name + " = '" + value +
                                 "' is not a whole number");
    }
}

}  // namespace

GPUSpec parse_spec(const std::string& text)
{
    GPUSpec spec;
    std::istringstream lines(text);
    std::string line;
    while (std::getline(lines, line)) {
        const std::string body = trimmed(line.substr(0, line.find('#')));
        if (body.empty()) {
            continue;
        }

        const size_t equals = body.find('=');
        if (equals == std::string::npos) {
            throw std::runtime_error("machine spec: '" + body + "' is not name = value");
        }
        const std::string name = trimmed(body.substr(0, equals));
        const std::string value = trimmed(body.substr(equals + 1));

        if (const char* why = fixed_field(name)) {
            throw std::runtime_error("machine spec: " + name +
                                     " is fixed, not configured — " + why);
        }

        const uint64_t number = to_number(name, value);
        if (name == "sm_count") {
            spec.sms.sm_count = static_cast<uint32_t>(number);
        } else if (name == "blocks_per_sm") {
            spec.sms.blocks_per_sm = static_cast<uint32_t>(number);
        } else if (name == "warp_slots_per_sm") {
            spec.sms.warp_slots_per_sm = static_cast<uint32_t>(number);
        } else if (name == "shared_bytes_per_sm") {
            spec.sms.shared_bytes_per_sm = static_cast<size_t>(number);
        } else if (name == "l1_lines") {
            spec.l1_lines = static_cast<size_t>(number);
        } else if (name == "l2_lines") {
            spec.l2_lines = static_cast<size_t>(number);
        } else {
            throw std::runtime_error("machine spec: no field named " + name);
        }
    }
    return spec;
}

GPUSpec load_spec(const std::string& path)
{
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("machine spec: cannot open " + path);
    }
    std::ostringstream all;
    all << file.rdbuf();
    return parse_spec(all.str());
}
