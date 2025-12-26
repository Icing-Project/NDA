#include "plugins/PythonPluginBridge.h"
#include "audio/AudioBuffer.h"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>

using namespace nda;
using namespace std::chrono;

struct BenchmarkResult {
    double mean_us;
    double median_us;
    double min_us;
    double max_us;
    double stddev_us;
};

BenchmarkResult benchmarkPlugin(const std::string& pluginPath,
                               const std::string& pluginDir,
                               const std::string& pluginType,
                               int iterations = 1000) {
    auto plugin = std::make_shared<PythonPluginBridge>();
    
    if (!plugin->loadPlugin(pluginPath, pluginDir)) {
        std::cerr << "Failed to load plugin: " << pluginPath << std::endl;
        return {0, 0, 0, 0, 0};
    }
    
    if (!plugin->initialize()) {
        std::cerr << "Failed to initialize plugin: " << pluginPath << std::endl;
        return {0, 0, 0, 0, 0};
    }
    
    if (!plugin->start()) {
        std::cerr << "Failed to start plugin: " << pluginPath << std::endl;
        plugin->shutdown();
        return {0, 0, 0, 0, 0};
    }
    
    AudioBuffer buffer(2, 512);  // 2 channels, 512 frames (standard)
    std::vector<double> timings;
    timings.reserve(iterations);
    
    // Warmup: 100 iterations
    for (int i = 0; i < 100; ++i) {
        if (pluginType == "source") {
            plugin->readAudio(buffer);
        } else if (pluginType == "processor") {
            plugin->processAudio(buffer);
        } else if (pluginType == "sink") {
            plugin->writeAudio(buffer);
        }
    }
    
    // Actual benchmark
    for (int i = 0; i < iterations; ++i) {
        auto start = high_resolution_clock::now();
        
        if (pluginType == "source") {
            plugin->readAudio(buffer);
        } else if (pluginType == "processor") {
            plugin->processAudio(buffer);
        } else if (pluginType == "sink") {
            plugin->writeAudio(buffer);
        }
        
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        timings.push_back(static_cast<double>(duration.count()));
    }
    
    plugin->stop();
    plugin->shutdown();
    
    // Calculate statistics
    BenchmarkResult result;
    
    double sum = 0.0;
    for (double t : timings) {
        sum += t;
    }
    result.mean_us = sum / timings.size();
    
    std::sort(timings.begin(), timings.end());
    result.median_us = timings[timings.size() / 2];
    result.min_us = timings.front();
    result.max_us = timings.back();
    
    // Standard deviation
    double variance = 0.0;
    for (double t : timings) {
        variance += (t - result.mean_us) * (t - result.mean_us);
    }
    result.stddev_us = std::sqrt(variance / timings.size());
    
    return result;
}

void printResult(const std::string& testName, const BenchmarkResult& result, double target_us) {
    std::cout << "\n" << testName << ":\n";
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "  Mean:   " << result.mean_us << " µs\n";
    std::cout << "  Median: " << result.median_us << " µs\n";
    std::cout << "  Min:    " << result.min_us << " µs\n";
    std::cout << "  Max:    " << result.max_us << " µs\n";
    std::cout << "  StdDev: " << result.stddev_us << " µs\n";
    std::cout << "  Target: " << target_us << " µs\n";
    
    if (result.mean_us < target_us) {
        std::cout << "  Status: ✓ PASS (target achieved)\n";
    } else {
        std::cout << "  Status: ✗ FAIL (target missed by " 
                  << (result.mean_us - target_us) << " µs)\n";
    }
}

void printUsage(const char* progName) {
    std::cout << "Usage: " << progName << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --plugin PATH       Path to plugin file (default: sine_wave_source.py)\n";
    std::cout << "  --plugin-dir DIR    Plugin directory (default: plugins_py)\n";
    std::cout << "  --type TYPE         Plugin type: source|processor|sink (default: source)\n";
    std::cout << "  --iterations N      Number of iterations (default: 1000)\n";
    std::cout << "  --target TARGET     Target performance in µs (default: 500)\n";
    std::cout << "  --help              Show this help\n";
}

int main(int argc, char** argv) {
    // Default parameters
    std::string pluginPath = "sine_wave_source.py";
    std::string pluginDir = "plugins_py";
    std::string pluginType = "source";
    int iterations = 1000;
    double target = 500.0;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--plugin" && i + 1 < argc) {
            pluginPath = argv[++i];
        } else if (arg == "--plugin-dir" && i + 1 < argc) {
            pluginDir = argv[++i];
        } else if (arg == "--type" && i + 1 < argc) {
            pluginType = argv[++i];
        } else if (arg == "--iterations" && i + 1 < argc) {
            iterations = std::stoi(argv[++i]);
        } else if (arg == "--target" && i + 1 < argc) {
            target = std::stod(argv[++i]);
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }
    
    std::cout << "=== Python Bridge Performance Benchmark ===\n";
    std::cout << "Plugin: " << pluginPath << "\n";
    std::cout << "Type: " << pluginType << "\n";
    std::cout << "Buffer: 2 channels, 512 frames\n";
    std::cout << "Iterations: " << iterations << " (+ 100 warmup)\n";
    std::cout << "Target: <" << target << " µs\n";
    
    auto result = benchmarkPlugin(pluginPath, pluginDir, pluginType, iterations);
    
    if (result.mean_us == 0.0) {
        std::cerr << "Benchmark failed (plugin error)\n";
        return 1;
    }
    
    printResult("Benchmark Results", result, target);
    
    std::cout << "\n=== Summary ===\n";
    if (result.mean_us < target) {
        std::cout << "Overall: ✓ PASS\n";
        
        // Show improvement vs baseline (if better than baseline)
        double baseline = 3000.0;  // Conservative baseline estimate
        if (result.mean_us < baseline) {
            double improvement = baseline / result.mean_us;
            std::cout << "Improvement vs baseline: " << std::fixed << std::setprecision(1) 
                     << improvement << "x faster\n";
        }
        
        return 0;
    } else {
        std::cout << "Overall: ✗ FAIL\n";
        std::cout << "Optimization needed to reach target\n";
        return 1;
    }
}
