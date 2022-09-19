#pragma once
#ifndef __ED_EBGINE_BASE_H__
#define __ED_EBGINE_BASE_H__

#include <vector>
#include <string>
#include <map>

enum EngineRuntime {
    MNN_RUNTIME_CPU = 100,
    MNN_RUNTIME_AUTO,
    MNN_RUNTIME_CUDA,
    MNN_RUNTIME_OPENCL,
    MNN_RUNTIME_VULKAN
};

enum EnginePrecision {
    MNN_PRECISION_NORMAL = 100,
    MNN_PRECISION_HIGH,
    MNN_PRECISION_LOW
};

enum EnginePower {
    MNN_POWER_NORMAL = 100,
    MNN_POWER_HIGH,
    MNN_POWER_LOW
};

enum EngineMemory {
    MNN_MEMORY_NORMAL = 100,
    MNN_MEMORY_HIGH,
    MNN_MEMORY_LOW
};

struct EdNodeInfo {
    /* data */
    int dataType;
    const char *name;
    size_t shape[4];
};

struct EdModelInfo {
    /* data */
    std::string modelPath;
    EngineRuntime runtime;
    EnginePrecision precision = MNN_PRECISION_NORMAL;
    EnginePower power;
    EngineMemory memory = MNN_MEMORY_NORMAL;
    std::vector<EdNodeInfo> inputNode;
    std::vector<EdNodeInfo> outputNode;
};


struct EdBuffer {
    /* data */
    void *bufferData;
    size_t bufferSize;
    int dataType;
};


#endif
