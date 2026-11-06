#include "strata/core/persistent_job.hpp"
#include <atomic>
#include <cassert>
#include <future>
#include <iostream>

static std::atomic<int> destroyed{0};
struct ThreadState { ~ThreadState() { ++destroyed; } };

int main() {
    int total = 0;
    std::thread::id worker_id;
    {
        strata::core::detail::PersistentJob worker;
        worker.wait(); // no job yet
        for (int i = 1; i <= 1000; ++i) {
            worker.post([&, i] {
                thread_local ThreadState state;
                if (i == 1) worker_id = std::this_thread::get_id();
                assert(worker_id == std::this_thread::get_id());
                assert(destroyed == 0); // TLS remains alive across every submission
                total += i;
            });
        }
        worker.wait();
        assert(total == 500500);
        std::promise<void> entered, release;
        auto done = release.get_future();
        worker.post([&] { entered.set_value(); done.wait(); ++total; });
        entered.get_future().wait();
        release.set_value();
        worker.wait();
        worker.post([&] { ++total; }); // destructor must drain the last job
    }
    assert(total == 500502);
    assert(destroyed == 1);
    { strata::core::detail::PersistentJob empty; }
    std::cout << "PASS: worker reuse, TLS lifetime, completion visibility and shutdown\n";
}
