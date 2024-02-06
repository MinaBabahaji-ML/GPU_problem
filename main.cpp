#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"

#include <algorithm>
#include <vector>
#include <random>
#include <iostream>
#include <cassert>
#include "cxxopts.hpp"

std::vector<float> openCL_inference(const char *model_path, std::vector<float> const &randomInput, int outputLength)
{
    TfLiteGpuDelegateOptionsV2 opts = TfLiteGpuDelegateOptionsV2Default();
    opts.is_precision_loss_allowed = 1;
    opts.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;
    opts.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
    opts.inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
    opts.inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO;

    TfLiteDelegate *gpuDelegate = TfLiteGpuDelegateV2Create(&opts);
    TfLiteInterpreterOptions *options = TfLiteInterpreterOptionsCreate();

    TfLiteInterpreterOptionsAddDelegate(options, gpuDelegate);

    TfLiteModel *model = TfLiteModelCreateFromFile(model_path);
    TfLiteInterpreter *interpreter = TfLiteInterpreterCreate(model, options);

    TfLiteInterpreterAllocateTensors(interpreter);
    auto *inputTensor = TfLiteInterpreterGetInputTensor(interpreter, 0);

    auto status = TfLiteTensorCopyFromBuffer(inputTensor, randomInput.data(), randomInput.size() * sizeof(float));
    assert(status == kTfLiteOk);

    TfLiteInterpreterInvoke(interpreter);

    std::vector<float> output(outputLength);
    auto const *outputTensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);

    status = TfLiteTensorCopyToBuffer(outputTensor, output.data(), output.size() * sizeof(float));
    assert(status == kTfLiteOk);

    TfLiteInterpreterDelete(interpreter);
    TfLiteGpuDelegateV2Delete(gpuDelegate);
    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(model);

    return output;
}

std::vector<float> xnnpack_inference(const char *model_path, std::vector<float> const &randomInput, int outputLength)
{
    TfLiteXNNPackDelegateOptions opts = TfLiteXNNPackDelegateOptionsDefault();
    opts.num_threads = 4;
    TfLiteDelegate *xnnpackDelegate = TfLiteXNNPackDelegateCreate(&opts);

    TfLiteInterpreterOptions *options = TfLiteInterpreterOptionsCreate();

    TfLiteInterpreterOptionsAddDelegate(options, xnnpackDelegate);

    TfLiteModel *model = TfLiteModelCreateFromFile(model_path);
    TfLiteInterpreter *interpreter = TfLiteInterpreterCreate(model, options);

    TfLiteInterpreterAllocateTensors(interpreter);
    auto *inputTensor = TfLiteInterpreterGetInputTensor(interpreter, 0);

    auto status = TfLiteTensorCopyFromBuffer(inputTensor, randomInput.data(), randomInput.size() * sizeof(float));
    assert(status == kTfLiteOk);

    TfLiteInterpreterInvoke(interpreter);

    std::vector<float> output(outputLength);
    auto const *outputTensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);

    status = TfLiteTensorCopyToBuffer(outputTensor, output.data(), output.size() * sizeof(float));
    assert(status == kTfLiteOk);

    TfLiteInterpreterDelete(interpreter);
    TfLiteXNNPackDelegateDelete(xnnpackDelegate);
    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(model);

    return output;
}

int main(int argc, char **argv)
{

    cxxopts::Options options("ModelTest", "Test model on mobile device");

    options.add_options()("a,model", "model name ", cxxopts::value<std::string>())("o,output_shape", "height, width and channel of output", cxxopts::value<std::vector<int>>())("i,input_shape", "height, width and channel of input", cxxopts::value<std::vector<int>>());
    ;

    auto result = options.parse(argc, argv);

    if (!result.count("model"))
    {
        throw std::runtime_error("You must provide model name.");
    }
    if (!result.count("input_shape") || !result.count("output_shape"))
    {
        throw std::runtime_error("You must provide input and output shapes.");
    }
    const std::vector<int> inputShape = result["input_shape"].as<std::vector<int>>();
    const std::vector<int> outputShape = result["output_shape"].as<std::vector<int>>();

    auto model_a = "./" + result["model"].as<std::string>();

    std::vector<float> randomInput(inputShape.at(0) * inputShape.at(1));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.f, 1.f);
    std::generate(randomInput.begin(), randomInput.end(), [&]()
                  { return dis(gen); });

    auto output_openCL = openCL_inference(const_cast<char *>(model_a.c_str()), randomInput, outputShape.at(0) * outputShape.at(1));

    auto output_xnnpack = xnnpack_inference(const_cast<char *>(model_a.c_str()), randomInput, outputShape.at(0) * outputShape.at(1));

    std::cout << "OpenCL output:" << std::endl;
    for (auto v : output_openCL)
        std::cout << v << ", ";
    std::cout << std::endl;

    std::cout << "xnnpack output:" << std::endl;
    for (auto v : output_xnnpack)
        std::cout << v << ", ";
    std::cout << std::endl;
}