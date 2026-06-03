# Build stable-diffusion.cpp on Android (Termux) with Vulkan

This guide outlines the steps to compile this repository on an Android mobile device using the Vulkan backend.

To get started, install **Termux** from the Play Store: [Termux on Google Play](https://play.google.com/store/apps/details?id=com.termux&hl=en).

## Quick Install (Automated)
You can automatically install and run everything with a single command:

**For Vulkan (Faster GPU Inference):**
```bash
curl -LsSf https://raw.githubusercontent.com/Juste-Leo2/stable-diffusion.cpp/refs/heads/bonsai_dev/sd-mobile-vulkan.sh | tr -d '\r' | bash
```

**For CPU (If Vulkan is not supported):**
```bash
curl -LsSf https://raw.githubusercontent.com/Juste-Leo2/stable-diffusion.cpp/refs/heads/bonsai_dev/sd-mobile-CPU.sh | tr -d '\r' | bash
```


## 1. System Update
First, ensure your Termux packages are up-to-date:
```bash
pkg update -y && pkg upgrade -y
```

## 2. Install Required Tools
Install the necessary dependencies for building:
```bash
pkg install -y clang cmake git ninja vulkan-headers vulkan-loader-generic shaderc python
pkg install -y spirv-headers
```

## 3. Clone Repository and Checkout Branch
Clone the repository and switch to the development branch:
```bash
git clone https://github.com/Juste-Leo2/stable-diffusion.cpp
cd stable-diffusion.cpp
git checkout bonsai_dev
```

## 4. Build Configuration
Create a build directory and configure the project with CMake:
```bash
mkdir build && cd build

cmake .. \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSD_VULKAN=ON \
  -DGGML_VULKAN_EMBED_SHADERS=ON \
  -DVulkan_LIBRARY=/system/lib64/libvulkan.so \
  -DVulkan_INCLUDE_DIR=$PREFIX/include \
  -DVulkan_GLSLC_EXECUTABLE=$PREFIX/bin/glslc
```

Compile the project using Ninja:
```bash
ninja
```

## 5. Prepare Models and Outputs Directory
Navigate back to the root directory and create the necessary folders:
```bash
cd ..
mkdir -p models outputs
```

## 6. Download Models

### Download Bonsai Model
```bash
curl -L "https://huggingface.co/Green-Sky/bonsai-image-binary-4B-GGUF/resolve/main/bonsai_image_4b-q1_0.gguf?download=true" -o models/bonsai_image_4b-q1_0.gguf
```

### Download VAE
```bash
curl -L "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors" -o models/flux2-vae.safetensors
```

### Download Qwen3 LLM
```bash
curl -L "https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-UD-Q3_K_XL.gguf?download=true" -o models/Qwen3-4B-UD-Q3_K_XL.gguf
```

## 7. Run Inference
Use the compiled binary to run the model:
```bash
./build/bin/sd-cli \
  --diffusion-model models/bonsai_image_4b-q1_0.gguf \
  --vae models/flux2-vae.safetensors \
  --llm models/Qwen3-4B-UD-Q3_K_XL.gguf \
  --cfg-scale 1 \
  --width 512 \
  --height 512 \
  --steps 6 \
  --seed 42 \
  -p "A cat is strolling through a park; the sun is shining; the cat is alone; in the background, there is a magnificent mountain landscape" \
  -o outputs/cat_test.png \
  --mmap \
  --fa
```