#include <torch/script.h>
#include <iostream>

class setParameters
{
private:
    torch::jit::script::Module torchModel;
    torch::Device device;
public:
    setParameters(torch::jit::script::Module torchModel);
    ~setParameters();

    // Inference
    at::Tensor Inference(torch::Tensor inputs);
};