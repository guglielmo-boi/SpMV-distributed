#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include <string>
#include <vector>
#include <filesystem>

inline std::vector<std::string> get_mtx_files(const std::string& data_dir) {
    std::vector<std::string> files;

    for (const auto& entry : std::filesystem::directory_iterator(data_dir)) {
        if (entry.path().extension() == ".mtx") {
            files.push_back(entry.path().string());
        }
    }

    return files;
}

#endif