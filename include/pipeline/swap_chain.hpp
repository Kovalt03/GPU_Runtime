#pragma once

#include <cstdint>
#include <vector>

#include "math3d.hpp"
#include "pipeline/draw.hpp"
#include "runtime.hpp"

// Two frames, so that the one being drawn is not the one being read.
//
// With a single frame a loop has to run draw, read, clear, draw — and the clear
// cannot start until the read has finished with the pixels it is about to
// destroy. With two, the clear of the next frame has nothing to do with the
// frame being drawn, so it can be queued on another stream and cost nothing.
//
// That overlap is the whole of what this buys here, and it is the first time
// streams reach real scene work: the draw routes are synchronous, so until
// something else had work to run beside them there was nothing to overlap with.
// The reading back is a host call and takes no device time at all — this models
// a swap chain's structure and the device half of its cost, not the display.
class SwapChain {
public:
    // Two frames of the target's size. More would change nothing here: with the
    // present being a host call, a third buffer has nothing further to hide.
    SwapChain(MyGPURuntime& rt, const DrawTarget& target);
    ~SwapChain();

    SwapChain(const SwapChain&) = delete;
    SwapChain& operator=(const SwapChain&) = delete;

    // The frame to draw into.
    const DeviceFrame& back() const;

    // Reads the back buffer out, makes the other one the back buffer, and clears
    // it ready to be drawn into.
    //
    // The clear is queued on a stream of its own when overlap is asked for, so it
    // runs beside the next draw rather than before it. Whoever draws next drains
    // the queue, which is what makes a synchronous draw route enough.
    std::vector<Float3> present(Float3 colour = Float3{0.0f, 0.0f, 0.0f},
                                float depth = 2.0f, bool overlap = true);

private:
    MyGPURuntime& rt_;
    DrawTarget target_;
    DeviceFrame frames_[2];
    uint32_t back_ = 0;
    StreamId clear_stream_ = DEFAULT_STREAM;
};
