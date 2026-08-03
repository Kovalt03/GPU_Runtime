#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <vector>

// Direction of a MemoryManager::memcpy. Naming the direction explicitly is what
// makes an address-space mix-up detectable instead of silently corrupting data.
enum class Direction { HostToDevice, DeviceToHost };

// One span of a managed buffer. Blocks tile their buffer with no gaps and in
// ascending offset order, so a freed block only ever has to be merged with its
// immediate neighbours.
struct Block {
    size_t offset;  // byte offset into the owning buffer
    size_t size;    // block size in bytes
    bool free;      // true while the block is available for allocation
};

// Simulates two completely separate address spaces: host (CPU) and device
// (GPU). Each is a flat byte buffer carved up by a first-fit free list.
//
// Pointers handed out are raw pointers into the owning std::vector, so neither
// buffer may ever be resized after construction — doing so would reallocate and
// dangle every outstanding pointer. Both are sized once in the constructor.
class MemoryManager {
public:
    // Allocation sizes are rounded up to this boundary. Real GPU allocators
    // align far more coarsely (256 B on NVIDIA); 16 is enough here to keep any
    // float or vec4 load naturally aligned.
    static constexpr size_t ALLOC_ALIGNMENT = 16;

    explicit MemoryManager(size_t host_size, size_t device_size);

    // Outstanding pointers refer into this instance's buffers, so copying would
    // hand out pointers owned by another manager.
    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;

    // --- Allocation ---------------------------------------------------------
    // Both throw std::runtime_error when size is 0 or no block is large enough.
    // Freeing nullptr is a no-op; freeing anything else this manager did not
    // hand out throws.
    void* host_alloc(size_t size);
    void host_free(void* ptr);

    void* device_alloc(size_t size);
    void device_free(void* ptr);

    // --- Transfer -----------------------------------------------------------
    // HostToDevice: dst must lie inside device memory, src must not.
    // DeviceToHost: src must lie inside device memory, dst must not.
    // The host side may be any host pointer, not just one from host_alloc —
    // this mirrors cudaMemcpy, which accepts an ordinary stack or vector
    // address. Violations and out-of-bounds spans throw std::runtime_error.
    void memcpy(void* dst, const void* src, size_t size, Direction dir);

    // --- Introspection (mainly for tests and statistics) --------------------
    size_t host_free_bytes() const;
    size_t device_free_bytes() const;

    // Largest single allocatable block. Compared against *_free_bytes() this
    // exposes fragmentation directly, which is how coalescing is tested.
    size_t device_largest_free_block() const;

    bool is_host_ptr(const void* ptr) const;
    bool is_device_ptr(const void* ptr) const;

private:
    // Host and device differ only in which buffer they carve up, so the
    // allocator logic lives here once rather than being duplicated per space.
    struct Arena {
        std::vector<uint8_t> bytes;
        // Holds used *and* free blocks: coalescing needs to see neighbours.
        // std::list because splitting and merging insert and erase in the
        // middle; the cache cost is irrelevant at simulator allocation counts.
        std::list<Block> blocks;
    };

    Arena host_;
    Arena device_;

    static void* alloc_from(Arena& arena, size_t size);
    static void free_from(Arena& arena, void* ptr, const char* space_name);
    static void coalesce(Arena& arena);
    static size_t free_bytes(const Arena& arena);
    static bool contains(const Arena& arena, const void* ptr);

    // Bounds-checks that [ptr, ptr + size) lies entirely within the arena.
    static void require_in_range(const Arena& arena, const void* ptr, size_t size,
                                 const char* what);
};
