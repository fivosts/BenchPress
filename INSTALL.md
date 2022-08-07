# Installation

### TLDR

`git clone` and `cd` into the repository and run:

```
mkdir build; cd build
cmake ..
make -j
```

### Prerequisites [Optional]

To install __BenchPress__ you need to have `cmake>=3.14` installed, and a few other standard apt packages. Inspect `requirements.apt` file to see what is required. Either `apt install` the listed packages or execute the script with sudo rights to install them. This step is optional.

### Python

__BenchPress__ uses `3.6<=python.version<=3.8`. `python3.8` is recommended. You may also use `python3.9` but pip will struggle to find older package versions. You will have to manually bump package versions in `requirements.pip`.

### Building BenchPress

Build makefiles:

```
cmake -S <path_to_src_root> -B <path_to_build_root> <-Dcmake_flag1, -Dcmake_flag2, ...>
```
For most of __BenchPress's__ functionalities (e.g. training and sampling the model), no cmake flags are required. The following build flags are supported:

- `-DLOCAL=<path>[Default: ""]`: All output binaries and libraries will be compiled under `<build_root>/local/`. Override this flag if you want to specify a custom output path (e.g. in the case of high-bandwidth partitions of clusters).
- `-DBOOST_FROM_SOURCE=ON/OF[Default: OFF]`: Set `ON` if you with boost library to be compiled from source into the build directory. __BenchPress__ does not use boost, however if you want to extend it to C++ synthesis and want boost header files to be visible by the language model, you can use this flag.
- `-DPROTOC_FROM_SOURCE=ON/OFF[Default: OFF]`: __BenchPress__ uses protobuf messages to read/write specifications about the model, corpuses, sampler and other things. To compiler protobufs `protoc` is needed. If you cannot install it globally, this flag will install it from source within the build directory.
- `-DBUILD_CLDRIVE=ON/OFF[Default: OFF]`: Set `ON` to build `cldrive`, a driver for OpenCL kernels. `cldrive` is required if one desires to execute kernels using __BenchPress's__ cli. Details: https://github.com/ChrisCummins/cldrive
- `-DBUILD_CSMITH=ON/OFF[Default: OFF]`: Builds `csmith` fuzzer and adds it to environment.
- `-DBUILD_CLSMITH=ON/OFF[Default: OFF]`: Builds `clsmith` fuzzer (`csmith` variation for OpenCL) and adds it to environment.
- `-DBUILD_MUTEC=ON/OFF[Default: OFF]`: Builds `mutec` source code mutator and adds it to __BenchPress's__ environment. Details: https://github.com/chao-peng/mutec
- `-DBUILD_SRCIROR=ON/OFF[Default: OFF]`: Builds `srciror` text-level and IR mutator and adds it to the environment. Details: https://github.com/TestingResearchIllinois/srciror

After you specify the build environment, `cd` into your build directory and:
```
make -j
```
This will produce all libraries and binaries sandboxed in the build directory. At the root of the source directory an executable script `benchpress` will be built:
```
./benchpress --help
```
will list all available execution flags for the application.

### Platforms

⚠️ __BenchPress__ only supports Linux-based systems. While it can be installed on Windows or MacOS, it has not been attempted and tested, therefore not guaranteed.
