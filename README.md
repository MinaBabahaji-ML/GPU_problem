# openCL delegate issue with a sequence of Dense/FullyConnected nodes 

This repo contains scripts and a tool to reproduce the `openCL` delegate issue with a sequence of Dense/FullyConnected nodes. Our experiments revealed that if we use a sequence of Dense layers in a special pattern (see the following image), the corresponding tflite version of this model will generate a bunch of `nan` and `inf` values for certain random indices in certain runs. This issue happens with both FP16 and FP32 tflite versions. This issue can't be reproduced with the `XNNPACK` delegate. And also this issue is for the trained or partially trained models and we could not regenrate issue with dummy model. 

<img width="153" alt="Screenshot 2023-01-16 at 2 43 50 PM" src="https://user-images.githubusercontent.com/45400368/212793054-8a85b2af-3a8b-47ee-9c90-9e8d58247f4a.png">

## Converting the model
* `model_files` folder contains the above-mentioned pattern (`sample.h5`) and its corresponding tflite version (`sample.tflite`). 
  * You can also use `convert_model.py` to convert this pattern to tflite.
  
  Note: `sample.h5` is extracted from a large trained model.

## tflite_inference tool 
We have implemented a small tool to feed a random input to our sample tflite model using `openCL` and `XNNPACK` delegates. Run the tool multiple times. You will see that the `openCL` delegate generates `nan` outputs while `XNNPACK` delegate generate values. 

### PREREQUISITES: ###
* Linux or Mac host computer
* Connectivity to the target device via adb
* Android NDK, version 22 or later
* CMake 3.18 or later

### BUILD INSTRUCTIONS ###
* Unzip the `tensorflow_lite_c_2_15_0.zip` file.
* In a terminal, from root folder:
```console
$ mkdir build
$ cd build
$cmake -G "Unix Makefiles" -DTensorFlowLiteC_ROOT=../tensorflow_lite_c_2_15_0 -DCMAKE_SYSTEM_NAME=Android -DANDROID_ABI=arm64-v8a -DANDROID_STL=c++_shared -DANDROID_NATIVE_API_LEVEL=26 -DCMAKE_VERBOSE_MAKEFILE=ON -DCMAKE_TOOLCHAIN_FILE=<path-to-ndk>/build/cmake/android.toolchain.cmake -DCMAKE_BUILD_TYPE=Release ..
$ make
```
* Here, you must replace <path-to-ndk> with the absolute path of the ndk installed on your computer. If you installed NDK through Android studio, it should be something similar to:
`$HOME/Android/ndk/25.1.8937393`

* `tensorflow_lite_c_2_15_0` is TensorflowFlow Lite library package.
### Run INSTRUCTIONS ###
WARNING: This step will write to your `/data/local/tmp` folder on device. Please make sure existing files in that folder are backed up as needed.

In a terminal, from root folder:
```console
$ adb push ./build/model_test /data/local/tmp
$ adb push ./model_files /data/local/tmp
$ adb push ./tensorflow_lite_c_2_15_0/lib/aarch64/libtensorflowlite_c.so /data/local/tmp
$ adb push ./tensorflow_lite_c_2_15_0/lib/aarch64/libtensorflowlite_gpu_delegate.so /data/local/tmp
```

To run the tool:
```console
$ adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=. ./model_test --model=model_files/sample.tflite --input_shape=1,105 --output_shape=1,78"
```

The output should be something like this:
```console
INFO: Created TensorFlow Lite delegate for GPU.
INFO: Initialized TensorFlow Lite runtime.
VERBOSE: Replacing 25 out of 25 node(s) with delegate (TfLiteGpuDelegateV2) node, yielding 1 partitions for the whole graph.
INFO: Initialized OpenCL-based API.
INFO: Created 1 GPU delegate kernels.
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
VERBOSE: Replacing 25 out of 25 node(s) with delegate (TfLiteXNNPackDelegate) node, yielding 1 partitions for the whole graph.
OpenCL output:
nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, 
xnnpack output:
-134169, 326967, 189859, 66076.4, 201469, 465716, -64249, 176900, -353304, -7664.85, -3.32638e+06, -195150, 58628.8, -3.40272e+06, -207893, 20605.6, -3.0792e+06, -128272, 36745.2, -2.85335e+06, -193326, 31585.5, -2.96961e+06, -157879, 89499.6, -2.557e+06, -196129, -25787.6, 481010, 175167, 74623.3, 502128, 180965, 242206, 401014, 195151, 269244, -20095.8, 342530, -240095, 12879.4, -342044, -154952, 2.34424e+06, 8829.61, 39918.2, 1.7395e+06, 18456.6, 148338, 1.76017e+06, 102718, -11030.9, 1.60906e+06, -45270.5, 116584, 1.77144e+06, -30711, -108513, 2.60172e+06, 199430, 1.06925e+06, 159435, -74301.6, -224833, 1.99951e+06, 99791.1, 93778.1, 1.93127e+06, 4493.21, 119425, 2.59004e+06, 153077, 1.07327e+06, 123002, -43977.9, -226290, 1.71432e+06, 95573.2, 
```
### Note ###
We have noticed that sometimes the above-mentioned pattern does not lead into wrong results. Therefore, we think in addition to the pattern structure, the values of weight and bias are also influential factors.

