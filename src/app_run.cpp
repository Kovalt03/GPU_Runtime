#include "app_run.hpp"

#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

uint32_t Args::number(size_t index, uint32_t fallback) const
{
    if (index >= positional.size()) {
        return fallback;
    }
    return static_cast<uint32_t>(std::atoi(positional[index].c_str()));
}

std::string Args::text(size_t index, const std::string& fallback) const
{
    return (index < positional.size()) ? positional[index] : fallback;
}

std::string Args::images_dir() const
{
    const std::string dir = out_dir + "images";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        throw std::runtime_error("cannot create " + dir + ": " + ec.message());
    }
    return dir + "/";
}

uint32_t Args::flag(const std::string& name, uint32_t fallback) const
{
    const auto found = flags.find(name);
    return (found == flags.end())
               ? fallback
               : static_cast<uint32_t>(std::atoi(found->second.c_str()));
}

Args parse_args(int argc, char** argv)
{
    Args args;
    std::string out = "benchmarks/result";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg.rfind("--", 0) == 0 && i + 1 < argc) {
            const std::string name = arg.substr(2);
            const std::string value = argv[++i];
            if (name == "out") {
                out = value;
            } else {
                args.flags[name] = value;
            }
        } else {
            args.positional.push_back(arg);
        }
    }

    std::error_code ec;
    std::filesystem::create_directories(out, ec);
    if (ec) {
        throw std::runtime_error("cannot create " + out + ": " + ec.message());
    }

    args.out_dir = out;
    if (!args.out_dir.empty() && args.out_dir.back() != '/') {
        args.out_dir.push_back('/');
    }
    return args;
}

std::string timestamped_run_dir()
{
    const std::time_t now = std::time(nullptr);
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif

    char stamp[32] = {};
    std::strftime(stamp, sizeof(stamp), "%Y-%m-%d_%H%M%S", &local);
    return std::string("benchmarks/result/") + stamp;
}
