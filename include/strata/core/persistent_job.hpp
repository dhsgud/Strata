#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>

namespace strata::core::detail {

// One owner submits non-throwing jobs. Reusing the thread retains its TLS I/O context.
class PersistentJob {
public:
    PersistentJob() : thread_([this] { loop(); }) {}
    ~PersistentJob() {
        wait();
        { std::lock_guard<std::mutex> lock(mutex_); stop_ = true; }
        cv_.notify_one();
        thread_.join();
    }
    PersistentJob(const PersistentJob&) = delete;
    PersistentJob& operator=(const PersistentJob&) = delete;
    void post(std::function<void()> job) {
        wait();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            job_ = std::move(job);
            busy_ = true;
        }
        cv_.notify_one();
    }
    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !busy_; });
    }
private:
    void loop() {
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            cv_.wait(lock, [this] { return busy_ || stop_; });
            if (stop_) return;
            auto job = std::move(job_);
            lock.unlock();
            job();
            lock.lock();
            busy_ = false;
            cv_.notify_all();
        }
    }
    std::mutex mutex_;
    std::condition_variable cv_;
    std::function<void()> job_;
    bool busy_ = false, stop_ = false;
    std::thread thread_;
};

} // namespace strata::core::detail
