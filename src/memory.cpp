#include <cstring>
#include <iterator>
#include <stdexcept>
#include <string>

#include "memory.hpp"

namespace {

// Rounding happens before the free-list search, never after: growing the size
// afterwards could select a block that no longer fits.
size_t align_up(size_t size, size_t alignment)
{
    if (size % alignment == 0) {
        return size;
    }
    return size - size % alignment + alignment;
}

}  // namespace

// ---------------------------------------------------------------------------
// Construction and introspection
// ---------------------------------------------------------------------------

MemoryManager::MemoryManager(size_t host_size, size_t device_size)
{
    // The one and only sizing of each buffer. Every pointer handed out later is
    // bytes.data() + offset, so resizing again would reallocate and dangle all
    // of them at once.
    host_.bytes.resize(host_size);
    device_.bytes.resize(device_size);

    // An arena starts as one free block covering everything; alloc_from splits
    // it from there. A zero-sized arena gets no block at all, so it reports 0
    // free bytes and every allocation against it fails.
    if (host_size > 0) {
        host_.blocks.push_back(Block{0, host_size, true});
    }
    if (device_size > 0) {
        device_.blocks.push_back(Block{0, device_size, true});
    }
}

bool MemoryManager::contains(const Arena& arena, const void* ptr)
{
    const uint8_t* base = arena.bytes.data();
    // Half-open: one-past-the-end belongs to whatever follows, not to us.
    return base <= ptr && ptr < base + arena.bytes.size();
}

size_t MemoryManager::free_bytes(const Arena& arena)
{
    size_t total = 0;
    for (const Block& block : arena.blocks) {
        if (block.free) {
            total += block.size;
        }
    }
    return total;
}

bool MemoryManager::is_host_ptr(const void* ptr) const
{
    return contains(host_, ptr);
}

bool MemoryManager::is_device_ptr(const void* ptr) const
{
    return contains(device_, ptr);
}

size_t MemoryManager::host_free_bytes() const
{
    return free_bytes(host_);
}

size_t MemoryManager::device_free_bytes() const
{
    return free_bytes(device_);
}

size_t MemoryManager::device_largest_free_block() const
{
    size_t largest = 0;
    for (const Block& block : device_.blocks) {
        if (block.free && block.size > largest) {
            largest = block.size;
        }
    }
    return largest;
}

// ---------------------------------------------------------------------------
// Allocation — first fit, with splitting
// ---------------------------------------------------------------------------

void* MemoryManager::alloc_from(Arena& arena, size_t size)
{
    if (size == 0) {
        throw std::runtime_error("alloc: size must be non-zero");
    }

    size = align_up(size, ALLOC_ALIGNMENT);

    for (auto it = arena.blocks.begin(); it != arena.blocks.end(); ++it) {
        if (!it->free || it->size < size) {
            continue;
        }

        const size_t remainder = it->size - size;
        if (remainder >= ALLOC_ALIGNMENT) {
            // The tail becomes a new free block immediately after this one,
            // which keeps blocks sorted by ascending offset — an ordering
            // coalesce() depends on.
            arena.blocks.insert(std::next(it), Block{it->offset + size, remainder, true});
            it->size = size;
        }
        // A remainder below the alignment is handed over with the block rather
        // than split off: a sliver nobody can allocate is worse than the waste.

        it->free = false;
        return arena.bytes.data() + it->offset;
    }

    // Nothing was modified on the way here, so a failed allocation leaves no
    // trace. The arena never grows — a real GPU has fixed VRAM, and growing
    // would reallocate bytes and dangle every outstanding pointer.
    throw std::runtime_error("alloc: out of memory");
}

void* MemoryManager::host_alloc(size_t size)
{
    return alloc_from(host_, size);
}

void* MemoryManager::device_alloc(size_t size)
{
    return alloc_from(device_, size);
}

// ---------------------------------------------------------------------------
// Deallocation — validate, mark, coalesce
// ---------------------------------------------------------------------------

void MemoryManager::free_from(Arena& arena, void* ptr, const char* space_name)
{
    if (ptr == nullptr) {
        return;  // Matches C's free(): freeing nothing is not an error.
    }
    if (!contains(arena, ptr)) {
        throw std::runtime_error(std::string(space_name) +
                                 ": pointer does not belong to this address space");
    }

    const size_t offset = static_cast<uint8_t*>(ptr) - arena.bytes.data();

    for (Block& block : arena.blocks) {
        // Equality, not containment: a pointer into the middle of a live
        // allocation must be rejected rather than free the whole block.
        if (block.offset != offset) {
            continue;
        }
        if (block.free) {
            throw std::runtime_error(std::string(space_name) + ": double free");
        }
        block.free = true;
        coalesce(arena);
        return;
    }

    throw std::runtime_error(std::string(space_name) +
                             ": not the start of an allocation");
}

void MemoryManager::host_free(void* ptr)
{
    free_from(host_, ptr, "host");
}

void MemoryManager::device_free(void* ptr)
{
    free_from(device_, ptr, "device");
}

// ---------------------------------------------------------------------------
// Coalescing
//
// Invariant: once this returns, no two neighbouring blocks are both free.
// Splitting only ever produces [used | free], so the only way to break the
// invariant is a free(), which is why every free() ends here.
// ---------------------------------------------------------------------------

void MemoryManager::coalesce(Arena& arena)
{
    if (arena.blocks.empty()) {
        return;  // std::next(begin()) would step past end() on an empty list.
    }

    auto it = arena.blocks.begin();
    while (std::next(it) != arena.blocks.end()) {
        auto next = std::next(it);
        if (it->free && next->free) {
            // Offsets are never rewritten: the leading block already starts
            // where the merged block starts.
            it->size += next->size;
            arena.blocks.erase(next);
            // Deliberately not advancing. Freeing a block between two free
            // neighbours produces a run of three, and advancing here would
            // merge only the first pair.
        } else {
            ++it;
        }
    }
}

// ---------------------------------------------------------------------------
// Transfer
// ---------------------------------------------------------------------------

void MemoryManager::require_in_range(const Arena& arena, const void* ptr, size_t size,
                                     const char* what)
{
    const uint8_t* base = arena.bytes.data();
    const uint8_t* p = static_cast<const uint8_t*>(ptr);
    if (p < base || p > base + arena.bytes.size()) {
        throw std::runtime_error(std::string(what) + ": pointer outside the arena");
    }

    // Compare against the space that remains rather than testing p + size:
    // a large size would overflow that sum and wrongly pass the check.
    const size_t available = static_cast<size_t>(base + arena.bytes.size() - p);
    if (size > available) {
        throw std::runtime_error(std::string(what) + ": transfer runs past the end");
    }
}

void MemoryManager::memcpy(void* dst, const void* src, size_t size, Direction dir)
{
    // Only the device side is bounds-checked. The host side may be any ordinary
    // address — a stack variable or a std::vector — whose extent this class has
    // no way to know, exactly as with cudaMemcpy. It only has to not be device
    // memory, which is what catches a direction used the wrong way round.
    if (dir == Direction::HostToDevice) {
        if (!is_device_ptr(dst)) {
            throw std::runtime_error(
                "memcpy: HostToDevice destination is not device memory");
        }
        if (is_device_ptr(src)) {
            throw std::runtime_error(
                "memcpy: HostToDevice source must not be device memory");
        }
        require_in_range(device_, dst, size, "memcpy dst");
    } else {
        if (!is_device_ptr(src)) {
            throw std::runtime_error("memcpy: DeviceToHost source is not device memory");
        }
        if (is_device_ptr(dst)) {
            throw std::runtime_error(
                "memcpy: DeviceToHost destination must not be device memory");
        }
        require_in_range(device_, src, size, "memcpy src");
    }

    // std::memcpy must stay qualified: an unqualified call resolves back to
    // this member function and recurses forever.
    std::memcpy(dst, src, size);
}
