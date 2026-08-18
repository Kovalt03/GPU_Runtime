#include "pipeline/swap_chain.hpp"

SwapChain::SwapChain(MyGPURuntime& rt, const DrawTarget& target)
    : rt_(rt), target_(target), clear_stream_(rt.myrt_stream_create())
{
    frames_[0] = allocate_frame(rt, target);
    frames_[1] = allocate_frame(rt, target);

    // Both start cleared, so that the first draw lands in a frame rather than in
    // whatever the allocation held. A depth-tested draw reads what is there.
    clear_frame(rt, frames_[0], target);
    clear_frame(rt, frames_[1], target);
}

SwapChain::~SwapChain()
{
    release(rt_, frames_[1]);
    release(rt_, frames_[0]);
}

const DeviceFrame& SwapChain::back() const
{
    return frames_[back_];
}

std::vector<Float3> SwapChain::present(Float3 colour, float depth, bool overlap)
{
    std::vector<Float3> pixels = read_back(rt_, frames_[back_]);
    back_ = 1 - back_;

    if (overlap) {
        // Queued rather than run. Nothing waits for it here: the next draw
        // drains the queue, and by then this has had a whole frame's work to
        // hide behind.
        queue_clear(rt_, frames_[back_], target_, colour, depth, clear_stream_);
    } else {
        clear_frame(rt_, frames_[back_], target_, colour, depth);
    }
    return pixels;
}
