#include <torch/script.h>
#include <iostream>

class cudaInference
{
private:
    torch::jit::script::Module torchModel;
    torch::Device device;
public:
    cudaInference(torch::jit::script::Module torchModel);
    ~cudaInference();

    // Inference
    at::Tensor Inference(torch::Tensor inputs);
};