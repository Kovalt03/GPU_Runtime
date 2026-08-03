#include <cstdint>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "memory.hpp"

namespace {

constexpr size_t HOST_SIZE = 4096;
constexpr size_t DEVICE_SIZE = 4096;

MemoryManager make_manager()
{
    return MemoryManager(HOST_SIZE, DEVICE_SIZE);
}

}  // namespace

// ---------------------------------------------------------------------------
// Allocation
// ---------------------------------------------------------------------------

TEST(Memory, AllocReturnsNonNull)
{
    MemoryManager mm = make_manager();

    void* p = mm.device_alloc(128);
    ASSERT_NE(p, nullptr);
    EXPECT_TRUE(mm.is_device_ptr(p));
    EXPECT_FALSE(mm.is_host_ptr(p)) << "device memory must not overlap host memory";
}

TEST(Memory, AllocationsDoNotOverlap)
{
    MemoryManager mm = make_manager();

    auto* a = static_cast<uint8_t*>(mm.device_alloc(100));
    auto* b = static_cast<uint8_t*>(mm.device_alloc(100));

    // Distinct allocations must not alias, or one kernel would silently
    // overwrite another's data.
    ASSERT_NE(a, b);
    const bool disjoint = (a + 100 <= b) || (b + 100 <= a);
    EXPECT_TRUE(disjoint) << "allocations overlap";
}

TEST(Memory, AllocationsAreAligned)
{
    MemoryManager mm = make_manager();

    // An odd size must still leave the next allocation aligned, otherwise a
    // float load in the ISA layer would land on an unaligned address.
    mm.device_alloc(1);
    void* p = mm.device_alloc(4);

    const auto addr = reinterpret_cast<uintptr_t>(p);
    EXPECT_EQ(addr % MemoryManager::ALLOC_ALIGNMENT, 0u);
}

TEST(Memory, FreeAndReallocSameSize)
{
    MemoryManager mm = make_manager();

    void* first = mm.device_alloc(256);
    const size_t after_alloc = mm.device_free_bytes();
    mm.device_free(first);

    EXPECT_GT(mm.device_free_bytes(), after_alloc) << "free must return bytes";

    void* second = mm.device_alloc(256);
    EXPECT_EQ(second, first) << "first-fit should reuse the block just freed";
}

TEST(Memory, FreeNullptrIsNoop)
{
    MemoryManager mm = make_manager();
    const size_t before = mm.device_free_bytes();

    EXPECT_NO_THROW(mm.device_free(nullptr));
    EXPECT_EQ(mm.device_free_bytes(), before);
}

TEST(Memory, DoubleFreeThrows)
{
    MemoryManager mm = make_manager();

    void* p = mm.device_alloc(64);
    mm.device_free(p);
    EXPECT_THROW(mm.device_free(p), std::runtime_error);
}

TEST(Memory, FreeUnknownPointerThrows)
{
    MemoryManager mm = make_manager();

    int stack_variable = 0;
    EXPECT_THROW(mm.device_free(&stack_variable), std::runtime_error);

    // A pointer into the middle of a live allocation is not a valid block start.
    auto* p = static_cast<uint8_t*>(mm.device_alloc(64));
    EXPECT_THROW(mm.device_free(p + 16), std::runtime_error);
}

TEST(Memory, ThrowOnZeroSize)
{
    MemoryManager mm = make_manager();
    EXPECT_THROW(mm.device_alloc(0), std::runtime_error);
}

TEST(Memory, ThrowOnOOM)
{
    MemoryManager mm = make_manager();
    EXPECT_THROW(mm.device_alloc(DEVICE_SIZE + 1), std::runtime_error);

    // Exhausting the arena must fail cleanly rather than hand back a bad block.
    void* whole = mm.device_alloc(DEVICE_SIZE);
    ASSERT_NE(whole, nullptr);
    EXPECT_THROW(mm.device_alloc(1), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Coalescing — the reason blocks live in a list rather than a bare free set
// ---------------------------------------------------------------------------

TEST(Memory, CoalesceAfterFree)
{
    MemoryManager mm = make_manager();

    void* a = mm.device_alloc(1024);
    void* b = mm.device_alloc(1024);
    void* c = mm.device_alloc(1024);
    void* d = mm.device_alloc(1024);  // consumes the tail, so no free space is
                                      // left adjacent to c

    // Freeing the outer two leaves two separate holes around a live block.
    mm.device_free(a);
    mm.device_free(c);
    EXPECT_LT(mm.device_largest_free_block(), 2048u)
        << "non-adjacent holes must not merge";

    // Freeing the middle one makes all three adjacent, so they must become a
    // single block big enough for an allocation none of them could satisfy.
    mm.device_free(b);
    EXPECT_GE(mm.device_largest_free_block(), 3072u);
    EXPECT_NO_THROW(mm.device_alloc(3072));

    mm.device_free(d);
}

TEST(Memory, FullCycleRestoresCapacity)
{
    MemoryManager mm = make_manager();
    const size_t initial = mm.device_free_bytes();

    std::vector<void*> ptrs;
    for (int i = 0; i < 8; ++i) {
        ptrs.push_back(mm.device_alloc(128));
    }
    for (void* p : ptrs) {
        mm.device_free(p);
    }

    // Repeated alloc/free must not leak capacity into stranded fragments.
    EXPECT_EQ(mm.device_free_bytes(), initial);
    EXPECT_EQ(mm.device_largest_free_block(), initial);
}

// ---------------------------------------------------------------------------
// Transfer
// ---------------------------------------------------------------------------

TEST(Memory, MemcpyHostToDevice)
{
    MemoryManager mm = make_manager();

    const std::vector<float> src = {1.0f, 2.0f, 3.0f, 4.0f};
    const size_t bytes = src.size() * sizeof(float);

    void* dev = mm.device_alloc(bytes);
    mm.memcpy(dev, src.data(), bytes, Direction::HostToDevice);

    // Read it straight back out to confirm the bytes actually landed.
    std::vector<float> readback(src.size(), 0.0f);
    mm.memcpy(readback.data(), dev, bytes, Direction::DeviceToHost);
    EXPECT_EQ(readback, src);
}

TEST(Memory, MemcpyDeviceToHost)
{
    MemoryManager mm = make_manager();

    const std::vector<uint8_t> pattern = {0xDE, 0xAD, 0xBE, 0xEF};
    void* dev = mm.device_alloc(pattern.size());
    mm.memcpy(dev, pattern.data(), pattern.size(), Direction::HostToDevice);

    std::vector<uint8_t> out(pattern.size(), 0);
    mm.memcpy(out.data(), dev, out.size(), Direction::DeviceToHost);
    EXPECT_EQ(out, pattern);
}

TEST(Memory, MemcpyAcceptsManagedHostMemory)
{
    MemoryManager mm = make_manager();

    // The host side may come from host_alloc as well as from ordinary memory.
    auto* host = static_cast<uint8_t*>(mm.host_alloc(4));
    host[0] = 0x11;
    host[1] = 0x22;
    host[2] = 0x33;
    host[3] = 0x44;

    void* dev = mm.device_alloc(4);
    mm.memcpy(dev, host, 4, Direction::HostToDevice);

    std::vector<uint8_t> out(4, 0);
    mm.memcpy(out.data(), dev, 4, Direction::DeviceToHost);
    EXPECT_EQ(out, (std::vector<uint8_t>{0x11, 0x22, 0x33, 0x44}));

    mm.host_free(host);
}

// ---------------------------------------------------------------------------
// Address-space separation — the core invariant of this layer
// ---------------------------------------------------------------------------

TEST(Memory, ThrowOnAddressMixup)
{
    MemoryManager mm = make_manager();

    void* dev = mm.device_alloc(64);
    std::vector<uint8_t> host(64, 0);

    // Direction says host→device, but the destination is host memory.
    EXPECT_THROW(mm.memcpy(host.data(), dev, 64, Direction::HostToDevice),
                 std::runtime_error);

    // Direction says device→host, but the source is not device memory.
    EXPECT_THROW(mm.memcpy(host.data(), host.data(), 64, Direction::DeviceToHost),
                 std::runtime_error);

    // Both ends in device memory is never a legal transfer in either direction.
    void* dev2 = mm.device_alloc(64);
    EXPECT_THROW(mm.memcpy(dev, dev2, 64, Direction::DeviceToHost), std::runtime_error);
}

TEST(Memory, HostAndDeviceAreDistinctSpaces)
{
    MemoryManager mm = make_manager();

    void* h = mm.host_alloc(64);
    void* d = mm.device_alloc(64);

    EXPECT_TRUE(mm.is_host_ptr(h));
    EXPECT_FALSE(mm.is_device_ptr(h));
    EXPECT_TRUE(mm.is_device_ptr(d));
    EXPECT_FALSE(mm.is_host_ptr(d));

    // Freeing a host pointer through the device allocator crosses the spaces.
    EXPECT_THROW(mm.device_free(h), std::runtime_error);
    EXPECT_THROW(mm.host_free(d), std::runtime_error);
}

TEST(Memory, MemcpyRejectsOutOfBounds)
{
    MemoryManager mm = make_manager();

    // Push the allocation off offset 0, so that a span starting at dev really
    // does run past the end instead of exactly filling the arena.
    mm.device_alloc(1024);
    void* dev = mm.device_alloc(64);
    std::vector<uint8_t> host(256, 0);

    // Starts inside device memory but runs past the end of the arena.
    EXPECT_THROW(mm.memcpy(dev, host.data(), DEVICE_SIZE, Direction::HostToDevice),
                 std::runtime_error);
}
