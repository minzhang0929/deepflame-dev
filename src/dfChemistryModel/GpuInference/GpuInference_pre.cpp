#include "GpuInference.H"

GpuInference::GpuInference() : device(torch::kCUDA) {}

GpuInference::GpuInference(torch::jit::script::Module torchModel)
    : torchModel(torchModel), device(torch::kCUDA)
{
    torchModel.to(device);
    std::cout << "load model successfully" << std::endl;
}

GpuInference::~GpuInference() {}

// Inference
at::Tensor GpuInference::Inference(torch::Tensor inputs)
{

    // torch::Tensor cudaInputs = inputs.to(device).toType(torch::kDouble);
    torch::Tensor cudaInputs = inputs.to(device);

    // inference and time monitor
    std::vector<torch::jit::IValue> INPUTS;
    INPUTS.push_back(cudaInputs);

    at::Tensor cudaOutput = torchModel.forward(INPUTS).toTensor();

    // printf("Outputs_F = %.10lf \n", cudaOutput[0][5].item().to<double>());

    // cudaOutput = cudaOutput.toType(torch::kDouble);
    return cudaOutput.to(torch::kCPU);
};
