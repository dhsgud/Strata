#pragma once

#if defined(__linux__)
#include <linux/aio_abi.h>
#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace strata::core::detail {

// One thread, one batch in flight. Keep a small private bounce buffer registered;
// the caller retains its existing completion validation and publication rules.
class FixedIoRing {
public:
    static constexpr size_t capacity = 6u << 20;
    static constexpr unsigned depth = 256;
    FixedIoRing() = default;
    FixedIoRing(const FixedIoRing&) = delete;
    FixedIoRing& operator=(const FixedIoRing&) = delete;
    ~FixedIoRing() { disable(); }

    bool open() {
        if (tried_) return fd_ >= 0;
        tried_ = true;
        fd_ = (int) syscall(SYS_io_uring_setup, depth, &p_);
        if (fd_ < 0) return failed();
        if (!(p_.features & IORING_FEAT_SINGLE_MMAP)) { errno = ENOTSUP; return failed(); }
        ring_size_ = std::max<size_t>(p_.sq_off.array + p_.sq_entries * sizeof(unsigned),
                                     p_.cq_off.cqes + p_.cq_entries * sizeof(io_uring_cqe));
        ring_ = mmap(nullptr, ring_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, IORING_OFF_SQ_RING);
        if (ring_ == MAP_FAILED) return failed();
        sqe_size_ = p_.sq_entries * sizeof(io_uring_sqe);
        sqes_ = mmap(nullptr, sqe_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, IORING_OFF_SQES);
        if (sqes_ == MAP_FAILED) return failed();
        buffer_ = mmap(nullptr, capacity, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (buffer_ == MAP_FAILED) return failed();
        (void) madvise(buffer_, capacity, MADV_NOHUGEPAGE);
        iovec v{buffer_, capacity};
        if (syscall(SYS_io_uring_register, fd_, IORING_REGISTER_BUFFERS, &v, 1) != 0) return failed();
        std::fprintf(stderr, "strata registered I/O: 6 MiB fixed buffer, io_uring enabled on the read worker\n");
        return true;
    }
    uint8_t* data() const { return (uint8_t*) buffer_; }

    // The supplied AIO descriptors already describe the validated, aligned windows.
    // A partial submission is discarded: only this ring's private buffer can still
    // be written, so callers can safely retry their unpublished destinations.
    long submit(iocb* const* cb, size_t n) {
        if (fd_ < 0 || n > p_.sq_entries) { errno = EINVAL; return -1; }
        auto* tail = word(p_.sq_off.tail);
        const unsigned from = __atomic_load_n(tail, __ATOMIC_RELAXED);
        if (__atomic_load_n(word(p_.sq_off.head), __ATOMIC_ACQUIRE) != from) { errno = EBUSY; return -1; }
        const unsigned mask = *word(p_.sq_off.ring_mask);
        auto* array = word(p_.sq_off.array);
        auto* sqe = (io_uring_sqe*) sqes_;
        for (size_t i = 0; i < n; ++i) {
            const iocb& c = *cb[i];
            const unsigned at = (from + (unsigned) i) & mask;
            io_uring_sqe& s = sqe[at];
            std::memset(&s, 0, sizeof s);
            s.opcode = IORING_OP_READ_FIXED;
            s.fd = (int) c.aio_fildes;
            s.off = (uint64_t) c.aio_offset;
            s.addr = c.aio_buf;
            s.len = (unsigned) c.aio_nbytes;
            s.user_data = c.aio_data;
            s.buf_index = 0;
            array[at] = at;
        }
        const unsigned end = from + (unsigned) n;
        __atomic_store_n(tail, end, __ATOMIC_RELEASE);
        for (;;) {
            const unsigned left = end - __atomic_load_n(word(p_.sq_off.head), __ATOMIC_ACQUIRE);
            if (left == 0) return (long) n;
            const long r = syscall(SYS_io_uring_enter, fd_, left, 0u, 0u, nullptr, 0u);
            if (r > 0 || (r < 0 && errno == EINTR)) continue;
            const int error = r == 0 ? EIO : errno;
            disable();
            errno = error;
            return -1;
        }
    }

    long reap(io_event* out, unsigned max, bool wait) {
        auto* head = word(p_.cq_off.head);
        unsigned from = __atomic_load_n(head, __ATOMIC_RELAXED);
        unsigned end = __atomic_load_n(word(p_.cq_off.tail), __ATOMIC_ACQUIRE);
        if (from == end) {
            // Enter even for a nonblocking poll: deferred completion task work may
            // need to run before the kernel publishes a CQE.
            if (syscall(SYS_io_uring_enter, fd_, 0u, wait ? 1u : 0u,
                        IORING_ENTER_GETEVENTS, nullptr, 0u) < 0) return -1;
            end = __atomic_load_n(word(p_.cq_off.tail), __ATOMIC_ACQUIRE);
        }
        const unsigned count = std::min(max, end - from), mask = *word(p_.cq_off.ring_mask);
        const auto* cqes = (const io_uring_cqe*) ((const uint8_t*) ring_ + p_.cq_off.cqes);
        for (unsigned i = 0; i < count; ++i) {
            const io_uring_cqe& c = cqes[(from + i) & mask];
            out[i] = {};
            out[i].data = c.user_data;
            out[i].res = c.res;
        }
        __atomic_store_n(head, from + count, __ATOMIC_RELEASE);
        return count;
    }

    void disable() {
        if (fd_ >= 0) close(fd_);
        fd_ = -1;
        if (sqes_ != MAP_FAILED) munmap(sqes_, sqe_size_);
        if (ring_ != MAP_FAILED) munmap(ring_, ring_size_);
        // A dedicated anonymous mapping, never malloc or caller-owned storage:
        // even if kernel teardown is asynchronous, its pinned physical pages cannot
        // be reused by a later allocation while an old read still references them.
        if (buffer_ != MAP_FAILED) munmap(buffer_, capacity);
        sqes_ = ring_ = buffer_ = MAP_FAILED;
    }
private:
    unsigned* word(unsigned offset) const { return (unsigned*) ((uint8_t*) ring_ + offset); }
    bool failed() {
        const int error = errno;
        disable();
        std::fprintf(stderr, "strata registered I/O: unavailable (%s); using Linux AIO\n", std::strerror(error));
        return false;
    }
    bool tried_ = false;
    int fd_ = -1;
    io_uring_params p_{};
    void* ring_ = MAP_FAILED;
    void* sqes_ = MAP_FAILED;
    void* buffer_ = MAP_FAILED;
    size_t ring_size_ = 0, sqe_size_ = 0;
};

} // namespace strata::core::detail
#endif
