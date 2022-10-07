#include "DNNInferencer.H"

DNNInferencer::DNNInferencer() : device_(torch::kCUDA) {}

DNNInferencer::DNNInferencer(torch::jit::script::Module torchModel, bool gpu)
    : torchModel_(torchModel), device_(torch::kCUDA), gpu_(gpu)
{
    at::TensorOptions opts;
    if (gpu_)
    {
        opts = at::TensorOptions().dtype(at::kDouble).device(at::kCUDA);
    }
    else
    {
        torch::Device device(at::kCPU);
        device_ = device;
        opts = at::TensorOptions().dtype(at::kDouble).device(at::kCPU);
    }
    torchModel_.to(device_);
    Xmu_vec = torch::tensor({1933.118541482812,
                             1.2327983023706526,
                             -5.705591538151852,
                             -6.446971251373195,
                             -4.169802387800032,
                             -6.1200334699867165,
                             -4.266343396329115,
                             -2.6007437468608616,
                             -0.4049762774428252},
                            opts);
    Xstd_vec = torch::tensor({716.6568054751183,
                              0.43268544913281914,
                              2.0857655247141387,
                              2.168997234412133,
                              2.707064105162402,
                              2.2681157746245897,
                              2.221785173612795,
                              1.5510851480805254,
                              0.30283229364455927},
                             opts);
    Ymu_vec = torch::tensor({175072.98234441387,
                             125434.41067566245,
                             285397.9376620931,
                             172924.8443087139,
                             -97451.53428068386,
                             -7160.953630852251,
                             -9.791262408691773e-10},
                            opts);
    Ystd_vec = torch::tensor({179830.51132577812,
                              256152.83860126554,
                              285811.9455262339,
                              263600.5448448552,
                              98110.53711881173,
                              11752.979335965118,
                              4.0735353885293555e-09},
                             opts);
    std::cout << "load model and parameters successfully" << std::endl;
}

DNNInferencer::~DNNInferencer() {}

// Inference
at::Tensor DNNInferencer::Inference(torch::Tensor inputs)
{
    torch::Tensor cudaInputs = inputs.to(device_);

    // generate tmpTensor
    auto Xmu_tensor = torch::unsqueeze(Xmu_vec, 0);
    auto Xstd_tensor = torch::unsqueeze(Xstd_vec, 0);
    auto Ymu_tensor = torch::unsqueeze(Ymu_vec, 0);
    auto Ystd_tensor = torch::unsqueeze(Ystd_vec, 0);

    // generate inputTensor
    torch::Tensor rhoInputs = torch::unsqueeze(cudaInputs.select(1, cudaInputs.sizes()[1] - 1), 1);
    torch::Tensor TInputs = torch::unsqueeze(cudaInputs.select(1, 0), 1);
    torch::Tensor pInputs = torch::unsqueeze(cudaInputs.select(1, 1) / 101325, 1);
    torch::Tensor YIndices = torch::linspace(2, cudaInputs.sizes()[1] - 2, cudaInputs.sizes()[1] - 3, device_).toType(torch::kLong);
    torch::Tensor YInputs = torch::index_select(cudaInputs, 1, YIndices);
    torch::Tensor YInputs_BCT = (torch::pow(YInputs, 0.1) - 1) / 0.1;

    torch::Tensor InfInputs = torch::cat({TInputs, pInputs, YInputs_BCT}, 1);
    InfInputs = (InfInputs - Xmu_tensor) / Xstd_tensor;

    InfInputs = InfInputs.toType(torch::kFloat);

    // inference and time monitor
    std::vector<torch::jit::IValue> INPUTS;
    INPUTS.push_back(InfInputs);

    at::Tensor cudaOutput = torchModel_.forward(INPUTS).toTensor();

    // generate outputTensor
    torch::Tensor deltaY = torch::index_select(cudaOutput, 1, YIndices);
    deltaY = deltaY * Ystd_tensor + Ymu_tensor;
    torch::Tensor Youtputs = torch::pow((YInputs_BCT + deltaY * 0.000001) * 0.1 + 1, 10);
    Youtputs = Youtputs / torch::sum(Youtputs, 1, 1);
    Youtputs = ((Youtputs - YInputs) * rhoInputs / 0.000001);

    return Youtputs;
};
