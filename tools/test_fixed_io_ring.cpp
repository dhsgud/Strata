#include "strata/core/fixed_io_ring.hpp"

#include <fcntl.h>
#include <sys/resource.h>
#include <array>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <vector>

static void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }

int main() {
    using strata::core::detail::FixedIoRing;
    char path[] = "strata-fixed-ring-test-XXXXXX";
    int output = mkstemp(path);
    require(output >= 0, "temporary file");
    struct Cleanup { const char* path; ~Cleanup() { unlink(path); } } cleanup{path};
    std::vector<uint8_t> bytes(FixedIoRing::capacity);
    uint64_t seed = 19;
    for (auto& b : bytes) { seed = seed * 6364136223846793005ull + 1; b = (uint8_t) (seed >> 56); }
    require(write(output, bytes.data(), bytes.size()) == (ssize_t) bytes.size(), "write fixture");
    close(output);
    int fd = open(path, O_RDONLY | O_DIRECT | O_CLOEXEC);
    require(fd >= 0, "open direct fixture");
    struct Close { int fd; ~Close() { close(fd); } } close_file{fd};
    FixedIoRing ring;
    require(ring.open(), "register buffer");
    constexpr size_t length = 512u << 10;
    auto batch = [&](bool eof, bool invalid_fd) {
        std::array<iocb, 2> cb{};
        std::array<iocb*, 2> ptr{&cb[0], &cb[1]};
        for (size_t i = 0; i < cb.size(); ++i) {
            cb[i].aio_fildes = (unsigned) (invalid_fd && i == 1 ? -1 : fd);
            cb[i].aio_lio_opcode = IOCB_CMD_PREAD;
            cb[i].aio_offset = (eof && i == 1) ? bytes.size() + 4096 : (i + 1) * length;
            cb[i].aio_buf = (uint64_t) (uintptr_t) (ring.data() + i * length);
            cb[i].aio_nbytes = length;
            cb[i].aio_data = i;
        }
        require(ring.submit(ptr.data(), ptr.size()) == 2, "submit batch");
        std::array<io_event, 2> events{};
        std::array<int, 2> seen{};
        unsigned done = 0;
        while (done < 2) {
            long n = ring.reap(events.data(), 2 - done, true);
            if (n < 0 && errno == EINTR) continue;
            require(n > 0, "reap batch");
            for (long j = 0; j < n; ++j) {
                const auto& e = events[(size_t) j];
                require(e.data < 2 && ++seen[(size_t) e.data] == 1, "completion identity");
                if (e.data == 1 && invalid_fd) require(e.res == -EBADF, "bad file result");
                else if (e.data == 1 && eof) require(e.res == 0, "short read result");
                else {
                    require(e.res == (int64_t) length, "complete read length");
                    require(std::memcmp(ring.data() + e.data * length,
                                        bytes.data() + (e.data + 1) * length, length) == 0, "exact read bytes");
                }
            }
            done += (unsigned) n;
        }
    };
    for (int i = 0; i < 32; ++i) batch(false, false);
    batch(true, false);
    batch(false, true);
    batch(false, false);
    io_event unused{};
    require(ring.reap(&unused, 1, false) == 0, "empty nonblocking completion");
    rlimit limit{};
    require(getrlimit(RLIMIT_MEMLOCK, &limit) == 0, "read memlock limit");
    rlimit low{0, limit.rlim_max};
    require(setrlimit(RLIMIT_MEMLOCK, &low) == 0, "lower local memlock limit");
    { FixedIoRing unavailable; require(!unavailable.open(), "registration must fail with zero allowance"); }
    require(setrlimit(RLIMIT_MEMLOCK, &limit) == 0, "restore local memlock limit");
    batch(false, false);
    void* address = ring.data();
    iocb last{};
    last.aio_fildes = (unsigned) fd;
    last.aio_lio_opcode = IOCB_CMD_PREAD;
    last.aio_buf = (uint64_t) (uintptr_t) address;
    last.aio_nbytes = bytes.size();
    iocb* last_ptr = &last;
    require(ring.submit(&last_ptr, 1) == 1, "submit before detach");
    ring.disable();
    void* guard = mmap(address, bytes.size(), PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
    require(guard == address, "reuse the detached virtual address");
    std::memset(guard, 0xa5, bytes.size());
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const auto* check = (const uint8_t*) guard;
    require(std::all_of(check, check + bytes.size(), [](uint8_t b) { return b == 0xa5; }),
            "old I/O modified a replacement anonymous mapping");
    munmap(guard, bytes.size());
    std::puts("PASS: fixed read bytes, reuse, short reads, bad fd, completion identity, memlock failure and in-flight detach");
}
