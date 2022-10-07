#ifndef DNNInferencer_H
#define DNNInferencer_H

#include <torch/script.h>
#include <iostream>
#include <memory>
#include "cuda_runtime.h"

class DNNInferencer
{
private:
    torch::jit::script::Module torchModel_;
    torch::Device device_;
    torch::Tensor Xmu_vec, Xstd_vec, Ymu_vec, Ystd_vec;
    bool gpu_;

public:
    DNNInferencer();
    DNNInferencer(torch::jit::script::Module torchModel, bool gpu);
    ~DNNInferencer();

    // Inference
    at::Tensor Inference(torch::Tensor inputs);
};

#endif