#ifndef GpuInference_H
#define GpuInference_H

#include <torch/script.h>
#include <iostream>
#include <memory>
#include "cuda_runtime.h"

class GpuInference
{
private:
    torch::jit::script::Module torchModel;
    torch::Device device;

public:
    GpuInference();
    GpuInference(torch::jit::script::Module torchModel);
    ~GpuInference();

    torch::Tensor Xmu_vec, Xstd_vec, Ymu_vec, Ystd_vec;

    // Inference
    at::Tensor Inference(torch::Tensor inputs);
};

#endif