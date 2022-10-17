#include "DNNInferencer.H"

DNNInferencer::DNNInferencer() : device_(torch::kCUDA) {}

DNNInferencer::DNNInferencer(torch::jit::script::Module torchModel)
    : torchModel_(torchModel), device_(torch::kCUDA)
{
    at::TensorOptions opts = at::TensorOptions().dtype(at::kDouble).device(at::kCUDA);
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

DNNInferencer::DNNInferencer(torch::jit::script::Module torchModel0, torch::jit::script::Module torchModel1,
    torch::jit::script::Module torchModel2, std::string device)
    : torchModel0_(torchModel0), torchModel1_(torchModel1), torchModel2_(torchModel2), device_(device)
{
    torchModel0_.to(device_);
    torchModel1_.to(device_);
    torchModel2_.to(device_);

    at::TensorOptions opts = at::TensorOptions().dtype(at::kDouble).device(device_);

    // set normalization parameters
    Xmu0_vec = torch::tensor({1282.4183173138097,
                            1.2539696320692386,
                            -7.667214603704226,
                            -9.006168975706206,
                            -8.837118398659516,
                            -2.3463333492248344,
                            -8.67517262679804,
                            -6.7342836663233285,
                            -7.482393463949848,
                            -9.43069358704131,
                            -9.536988287577792,
                            -7.386097302538013,
                            -2.1940688042993903,
                            -7.414727399905766,
                            -8.43683079918986,
                            -9.096284725537341,
                            -6.806847970901267,
                            -8.88567506884142,
                            -7.840187602975177,
                            -9.00125399892105,
                            -7.3377720480561885,
                            -3.0917924495344384},
                            opts);
    Xstd0_vec = torch::tensor({206.68816260797863,
                            0.43240469796322983,
                            1.2569012063271943,
                            0.3658696984801769,
                            0.6048604803125679,
                            2.1429685639707916,
                            0.6377329206030047,
                            1.809820414763561,
                            0.9148165943283916,
                            0.3593858969688066,
                            0.27646088853440465,
                            0.8605392540263817,
                            2.151418826571876,
                            1.50190901270806,
                            1.410807279887814,
                            0.5049010957724376,
                            1.6388487819838793,
                            0.4963069213619818,
                            1.4523705990139852,
                            0.557886263718149,
                            1.5690423335340304,
                            2.356414933319614},
                            opts);
    Ymu0_vec = torch::tensor({
                            4823.801521617341,
                            2727.12829776585,
                            4391.280672612407,
                            -2.6327050100454583,
                            4751.7876693798025,
                            6009.187993440942,
                            8420.668451269325,
                            2061.753135792586,
                            1789.6350500405217,
                            7410.485319581533,
                            -17.697301899440774,
                            4476.684923000168,
                            2193.397488064959,
                            3340.313244729177,
                            6823.083639788729,
                            3905.1729657311257,
                            4437.900773500976,
                            3441.7394569988255,
                            5253.7210797920025,
                            -3.400018618643355e-12},
                            opts);
    Ystd0_vec = torch::tensor({
                            39862.034376366784,
                            39587.83997268838,
                            45599.38276380988,
                            15.100583992983287,
                            46906.956121947725,
                            42517.889315103654,
                            69681.34452016829,
                            14489.407776141179,
                            12009.23413360539,
                            64053.51461616675,
                            107.70049760538379,
                            41144.40283698755,
                            15768.9000371634,
                            35057.39997863424,
                            41538.52879308667,
                            34641.123694890084,
                            24653.940719931026,
                            26867.219095839075,
                            27026.174250314187,
                            1.1235275712644957e-08},
                            opts);
    Xmu1_vec = torch::tensor({2060.8252779292293,
                            1.235733971577472,
                            -5.076962664954135,
                            -6.7133171181061115,
                            -7.577656895223432,
                            -5.998817299319277,
                            -6.885859995853336,
                            -4.740992879602225,
                            -8.247531007857225,
                            -7.717808576148032,
                            -8.24825243934851,
                            -6.566701906906986,
                            -5.729388004172249,
                            -4.085445789815756,
                            -5.824820539104126,
                            -8.05434066763203,
                            -7.357766350902313,
                            -8.957139637044149,
                            -5.025892745609114,
                            -7.901528169431469,
                            -7.499166073609093,
                            -3.202777083869064},
                            opts);
    Xstd1_vec = torch::tensor({519.0946800661171,
                            0.43406206020224564,
                            2.313407349506001,
                            1.4635122594184171,
                            1.8308382190076267,
                            2.9598944431116676,
                            1.7932118858862602,
                            2.840631122694604,
                            1.302430215158888,
                            1.5042142737058872,
                            1.1862062538742957,
                            1.8988518196451811,
                            2.8850439208839935,
                            2.5903920183832287,
                            2.4661899462408723,
                            0.9539300157341408,
                            1.5781358256928404,
                            0.7685720748926116,
                            3.334374836740926,
                            1.389140075720197,
                            1.95086915717926,
                            2.6424450267132347},
                            opts);
    Ymu1_vec = torch::tensor({
                            108017.47743342962,
                            103896.7956606026,
                            78322.46439090716,
                            17278.57189146481,
                            108638.85901645076,
                            115619.84568123752,
                            53950.721487655785,
                            35502.666488461145,
                            20080.378301073426,
                            110853.37857821377,
                            63331.25873261109,
                            125381.06475680208,
                            102561.69010479066,
                            29404.396336510683,
                            70977.12854400859,
                            -1704.3888959099413,
                            76281.36224895103,
                            26133.65582324682,
                            41808.39599404272,
                            -4.0787931227111724e-08},
                            opts);
    Ystd1_vec = torch::tensor({
                            495025.60004745435,
                            523148.718471993,
                            552136.9556846514,
                            224198.5655536522,
                            643007.4145062551,
                            571219.0622916721,
                            476729.0513066844,
                            455335.08968152356,
                            402887.5751804063,
                            669272.6716404198,
                            434083.16677684867,
                            607958.6094824811,
                            484550.90159023256,
                            495327.41290092416,
                            542710.2483693821,
                            362145.71260920743,
                            493632.2316900689,
                            453749.70536208263,
                            425332.51465037954,
                            8.151863349544284e-05},
                            opts);
    Xmu2_vec = torch::tensor({2967.6468301372443,
                            1.243683006639968,
                            -4.212556159758001,
                            -4.9155905064719185,
                            -4.800104015355045,
                            -4.795930983248498,
                            -4.429621650683212,
                            -3.3825206689586755,
                            -7.670046476519993,
                            -8.711212007184969,
                            -8.974952949087577,
                            -8.61232541853145,
                            -8.805005285024478,
                            -2.5649819138212755,
                            -4.184163238665646,
                            -7.902664543802336,
                            -8.48287621859309,
                            -9.487484315309668,
                            -8.914633729344454,
                            -9.555747208636628,
                            -9.679158904233338,
                            -2.7788791922721146},
                            opts);
    Xstd2_vec = torch::tensor({278.9810711441267,
                            0.4339953188226321,
                            1.9500160697992106,
                            1.1746554967709013,
                            1.6917771696211221,
                            2.471141134151887,
                            1.6745944040654246,
                            2.2727665385108367,
                            1.0782301065182132,
                            1.3244932550711974,
                            1.05062233833517,
                            1.4918007230086938,
                            1.3886059976275935,
                            2.0625556553283286,
                            1.9676977367664865,
                            0.6518606802545863,
                            0.7575317131821949,
                            0.42560013170241007,
                            1.9420735335037163,
                            0.8224050201147591,
                            0.6211629439697558,
                            2.39359977035721},
                            opts);
    Ymu2_vec = torch::tensor({
                            -28.80853287765839,
                            1841.7350192857177,
                            2531.0110238661587,
                            -7005.615026495482,
                            150.9943449800927,
                            -2398.4375581597264,
                            -7187.029992382052,
                            -19108.50617926657,
                            -15956.893851162014,
                            -23630.979827219977,
                            -18116.66147768638,
                            980.0095134555011,
                            7784.272315290288,
                            -12471.113866601243,
                            -25545.686697542653,
                            -12990.699558455324,
                            -17044.555122105623,
                            -12206.344135447025,
                            -8904.936176085748,
                            4.36006529397597e-09},
                            opts);
    Ystd2_vec = torch::tensor({
                            32321.05590450801,
                            19464.552370919984,
                            39048.52499795168,
                            51481.90282288761,
                            35314.09338501899,
                            36774.7824052581,
                            48398.803686861334,
                            95336.80885827845,
                            79968.39036741608,
                            106965.5377384096,
                            92435.05279354649,
                            18315.632630776836,
                            53733.78259330722,
                            78860.66429349894,
                            124678.71934163885,
                            60372.40300992112,
                            95814.61317076812,
                            57388.953767533836,
                            43986.93598552721,
                            7.135805283303832e-06},
                            opts);
    std::cout << "index = " << int(device_.index()) << std::endl;
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
}

std::vector<std::vector<double>> DNNInferencer::Inference_multiDNNs(std::vector<std::vector<double>> DNNinputs, int dimension)
{
    // generate tensor
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    torch::Tensor cudaInputs0 = torch::tensor(DNNinputs[0]).reshape({-1, dimension}).to(device_);
    torch::Tensor cudaInputs1 = torch::tensor(DNNinputs[1]).reshape({-1, dimension}).to(device_);
    torch::Tensor cudaInputs2 = torch::tensor(DNNinputs[2]).reshape({-1, dimension}).to(device_);

    // send inputs to GPU
    // torch::Tensor cudaInputs0 = inputs0.to(device_);
    // torch::Tensor cudaInputs1 = inputs1.to(device_);
    // torch::Tensor cudaInputs2 = inputs2.to(device_);

    // generate tmpTensor
    auto Xmu0_tensor = torch::unsqueeze(Xmu0_vec, 0);
    auto Xstd0_tensor = torch::unsqueeze(Xstd0_vec, 0);
    auto Ymu0_tensor = torch::unsqueeze(Ymu0_vec, 0);
    auto Ystd0_tensor = torch::unsqueeze(Ystd0_vec, 0);

    auto Xmu1_tensor = torch::unsqueeze(Xmu1_vec, 0);
    auto Xstd1_tensor = torch::unsqueeze(Xstd1_vec, 0);
    auto Ymu1_tensor = torch::unsqueeze(Ymu1_vec, 0);
    auto Ystd1_tensor = torch::unsqueeze(Ystd1_vec, 0);

    auto Xmu2_tensor = torch::unsqueeze(Xmu2_vec, 0);
    auto Xstd2_tensor = torch::unsqueeze(Xstd2_vec, 0);
    auto Ymu2_tensor = torch::unsqueeze(Ymu2_vec, 0);
    auto Ystd2_tensor = torch::unsqueeze(Ystd2_vec, 0);

    // normalization and BCT trans
    torch::Tensor rhoInputs0 = torch::unsqueeze(cudaInputs0.select(1, cudaInputs0.sizes()[1] - 1), 1);
    torch::Tensor TInputs0 = torch::unsqueeze(cudaInputs0.select(1, 0), 1);
    torch::Tensor pInputs0 = torch::unsqueeze(cudaInputs0.select(1, 1), 1);
    torch::Tensor Y_AR0 = torch::zeros_like(TInputs0);
    torch::Tensor YIndices = torch::linspace(2, cudaInputs0.sizes()[1] - 2, cudaInputs0.sizes()[1] - 3, device_).toType(torch::kLong);
    torch::Tensor YInputs0 = torch::index_select(cudaInputs0, 1, YIndices);
    torch::Tensor YInputs0_BCT = (torch::pow(YInputs0, 0.1) - 1) / 0.1;
    torch::Tensor InfInputs0 = torch::cat({TInputs0, pInputs0, YInputs0_BCT}, 1);
    InfInputs0 = (InfInputs0 - Xmu0_tensor) / Xstd0_tensor;
    InfInputs0 = torch::cat({InfInputs0, Y_AR0}, 1); // set Y_AR to 0

    torch::Tensor rhoInputs1 = torch::unsqueeze(cudaInputs1.select(1, cudaInputs1.sizes()[1] - 1), 1);
    torch::Tensor TInputs1 = torch::unsqueeze(cudaInputs1.select(1, 0), 1);
    torch::Tensor pInputs1 = torch::unsqueeze(cudaInputs1.select(1, 1), 1);
    torch::Tensor Y_AR1 = torch::zeros_like(TInputs1);
    torch::Tensor YInputs1 = torch::index_select(cudaInputs1, 1, YIndices);
    torch::Tensor YInputs1_BCT = (torch::pow(YInputs1, 0.1) - 1) / 0.1;
    torch::Tensor InfInputs1 = torch::cat({TInputs1, pInputs1, YInputs1_BCT}, 1);
    InfInputs1 = (InfInputs1 - Xmu1_tensor) / Xstd1_tensor;
    InfInputs1 = torch::cat({InfInputs1, Y_AR1}, 1); // set Y_AR to 0

    torch::Tensor rhoInputs2 = torch::unsqueeze(cudaInputs2.select(1, cudaInputs2.sizes()[1] - 1), 1);
    torch::Tensor TInputs2 = torch::unsqueeze(cudaInputs2.select(1, 0), 1);
    torch::Tensor pInputs2 = torch::unsqueeze(cudaInputs2.select(1, 1), 1);
    torch::Tensor Y_AR2 = torch::zeros_like(TInputs2);
    torch::Tensor YInputs2 = torch::index_select(cudaInputs2, 1, YIndices);
    torch::Tensor YInputs2_BCT = (torch::pow(YInputs2, 0.1) - 1) / 0.1;
    torch::Tensor InfInputs2 = torch::cat({TInputs2, pInputs2, YInputs2_BCT}, 1);
    InfInputs2 = (InfInputs2 - Xmu2_tensor) / Xstd2_tensor;
    InfInputs2 = torch::cat({InfInputs2, Y_AR2}, 1); // set Y_AR to 0
    
    std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now();
    std::chrono::duration<double> processingTime = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
    std::cout << "preInf time = " << processingTime.count() << std::endl;
    time_preInf += processingTime.count();

    // inference
    std::chrono::steady_clock::time_point start1 = std::chrono::steady_clock::now();

    InfInputs0 = InfInputs0.toType(torch::kFloat);
    std::vector<torch::jit::IValue> INPUTS0;
    INPUTS0.push_back(InfInputs0);
    at::Tensor cudaOutput0 = torchModel0_.forward(INPUTS0).toTensor();

    InfInputs1 = InfInputs1.toType(torch::kFloat);
    std::vector<torch::jit::IValue> INPUTS1;
    INPUTS1.push_back(InfInputs1);
    at::Tensor cudaOutput1 = torchModel1_.forward(INPUTS1).toTensor();

    InfInputs2 = InfInputs2.toType(torch::kFloat);
    std::vector<torch::jit::IValue> INPUTS2;
    INPUTS2.push_back(InfInputs2);
    at::Tensor cudaOutput2 = torchModel2_.forward(INPUTS2).toTensor();

    std::chrono::steady_clock::time_point stop1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> processingTime1 = std::chrono::duration_cast<std::chrono::duration<double>>(stop1 - start1);
    std::cout << "Inf time = " << processingTime1.count() << std::endl;
    time_Inference += processingTime1.count();

    // generate outputTensor
    std::chrono::steady_clock::time_point start2 = std::chrono::steady_clock::now();

    std::vector<std::vector<double>> results;

    torch::Tensor deltaY0 = torch::index_select(cudaOutput0, 1, YIndices);
    deltaY0 = deltaY0 * Ystd0_tensor + Ymu0_tensor;
    torch::Tensor Youtputs0 = torch::pow((YInputs0_BCT + deltaY0 * 0.000001) * 0.1 + 1, 10);
    Youtputs0 = Youtputs0 / torch::sum(Youtputs0, 1, 1);
    Youtputs0 = ((Youtputs0 - YInputs0) * rhoInputs0 / 0.000001);
    Youtputs0 = torch::cat({Youtputs0, Y_AR0}, 1);

    torch::Tensor deltaY1 = torch::index_select(cudaOutput1, 1, YIndices);
    deltaY1 = deltaY1 * Ystd1_tensor + Ymu1_tensor;
    torch::Tensor Youtputs1 = torch::pow((YInputs1_BCT + deltaY1 * 0.000001) * 0.1 + 1, 10);
    Youtputs1 = Youtputs1 / torch::sum(Youtputs1, 1, 1);
    Youtputs1 = ((Youtputs1 - YInputs1) * rhoInputs1 / 0.000001);
    Youtputs1 = torch::cat({Youtputs1, Y_AR1}, 1);

    torch::Tensor deltaY2 = torch::index_select(cudaOutput2, 1, YIndices);
    deltaY2 = deltaY2 * Ystd2_tensor + Ymu2_tensor;
    torch::Tensor Youtputs2 = torch::pow((YInputs2_BCT + deltaY2 * 0.000001) * 0.1 + 1, 10);
    Youtputs2 = Youtputs2 / torch::sum(Youtputs2, 1, 1);
    Youtputs2 = ((Youtputs2 - YInputs2) * rhoInputs2 / 0.000001);
    Youtputs2 = torch::cat({Youtputs2, Y_AR2}, 1);

    std::chrono::steady_clock::time_point start3 = std::chrono::steady_clock::now();

    Youtputs0 = Youtputs0.to(at::kCPU);
    Youtputs1 = Youtputs1.to(at::kCPU);
    Youtputs2 = Youtputs2.to(at::kCPU);

    std::chrono::steady_clock::time_point stop3 = std::chrono::steady_clock::now();
    std::chrono::duration<double> processingTime3 = std::chrono::duration_cast<std::chrono::duration<double>>(stop3 - start3);
    std::cout << "hot time = " << processingTime3.count() << std::endl;
    time_hot += processingTime3.count();

    std::vector<double> RRoutputs0(Youtputs0.data_ptr<double>(), Youtputs0.data_ptr<double>() + Youtputs0.numel());
    std::vector<double> RRoutputs1(Youtputs1.data_ptr<double>(), Youtputs1.data_ptr<double>() + Youtputs1.numel());
    std::vector<double> RRoutputs2(Youtputs2.data_ptr<double>(), Youtputs2.data_ptr<double>() + Youtputs2.numel());

    results = {RRoutputs0, RRoutputs1, RRoutputs2};

    std::chrono::steady_clock::time_point stop2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> processingTime2 = std::chrono::duration_cast<std::chrono::duration<double>>(stop2 - start2);
    std::cout << "postInf time = " << processingTime2.count() << std::endl;
    time_postInf += processingTime2.count();

    std::cout << "preInf sum time = " << time_preInf << std::endl;
    std::cout << "Inf sum time = " << time_Inference << std::endl;
    std::cout << "postInf sum time = " << time_postInf << std::endl;
    std::cout << "hot sum time = " << time_hot << std::endl;

    return results;
}
