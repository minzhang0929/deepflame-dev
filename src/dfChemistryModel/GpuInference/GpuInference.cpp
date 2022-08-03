#include "GpuInference.H"

GpuInference::GpuInference() : device(torch::kCUDA) {}

GpuInference::GpuInference(torch::jit::script::Module torchModel)
    : torchModel(torchModel), device(torch::kCUDA)
{
    torchModel.to(device);
    at::TensorOptions opts = at::TensorOptions().dtype(at::kDouble).device(at::kCUDA);
    // at::TensorOptions opts = at::TensorOptions().device(at::kCUDA);
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
    // printf("Ystd_vec = %.10lf \n", Ystd_vec[5].item().to<double>());
    // std::cout << Ystd_vec << std::endl;
    std::cout << "load model and parameters successfully" << std::endl;
}

GpuInference::~GpuInference() {}

// Inference
at::Tensor GpuInference::Inference(torch::Tensor inputs)
{

    // torch::Tensor cudaInputs = inputs.to(device).toType(torch::kDouble);
    torch::Tensor cudaInputs = inputs.to(device);

    // generate tmpTensor
    auto Xmu_tensor = torch::unsqueeze(Xmu_vec, 0);
    auto Xstd_tensor = torch::unsqueeze(Xstd_vec, 0);
    auto Ymu_tensor = torch::unsqueeze(Ymu_vec, 0);
    auto Ystd_tensor = torch::unsqueeze(Ystd_vec, 0);

    // generate inputTensor
    torch::Tensor rhoInputs = torch::unsqueeze(cudaInputs.select(1, cudaInputs.sizes()[1] - 1), 1);
    torch::Tensor TInputs = torch::unsqueeze(cudaInputs.select(1, 0), 1);
    torch::Tensor pInputs = torch::unsqueeze(cudaInputs.select(1, 1) / 101325, 1);
    torch::Tensor YIndices = torch::linspace(2, cudaInputs.sizes()[1] - 2, cudaInputs.sizes()[1] - 3, device).toType(torch::kLong);
    torch::Tensor YInputs = torch::index_select(cudaInputs, 1, YIndices);
    torch::Tensor YInputs_BCT = (torch::pow(YInputs, 0.1) - 1) / 0.1;

    torch::Tensor InfInputs = torch::cat({TInputs, pInputs, YInputs_BCT}, 1);
    InfInputs = (InfInputs - Xmu_tensor) / Xstd_tensor;

    // printf("Inputs_D = %.10lf \n", InfInputs[0][5].item().to<double>());

    InfInputs = InfInputs.toType(torch::kFloat);

    // printf("Inputs_F = %.10lf \n", InfInputs[0][5].item().to<double>());

    // inference and time monitor
    std::vector<torch::jit::IValue> INPUTS;
    INPUTS.push_back(InfInputs);

    at::Tensor cudaOutput = torchModel.forward(INPUTS).toTensor();

    // printf("Outputs_F = %.10lf \n", cudaOutput[0][5].item().to<double>());

    // cudaOutput = cudaOutput.toType(torch::kDouble);

    // printf("Outputs_D = %.10lf \n", cudaOutput[0][5].item().to<double>());

    // generate outputTensor
    torch::Tensor deltaY = torch::index_select(cudaOutput, 1, YIndices);
    deltaY = deltaY * Ystd_tensor + Ymu_tensor;
    torch::Tensor Youtputs = torch::pow((YInputs_BCT + deltaY * 0.000001) * 0.1 + 1, 10);
    // torch::Tensor Y_sum = torch::sum(Youtputs, 1, 1);
    Youtputs = Youtputs / torch::sum(Youtputs, 1, 1);
    Youtputs = ((Youtputs - YInputs) * rhoInputs / 0.000001);
    // printf("Outputs = %lf \n", Youtputs[0][5].item().to<double>());

    return Youtputs;
};
