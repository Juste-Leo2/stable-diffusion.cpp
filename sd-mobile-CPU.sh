#!/bin/bash

# Setup colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Stable-Diffusion.cpp Termux Setup (CPU) ===${NC}"

# Check if we are inside the stable-diffusion.cpp directory or need to clone
if [ ! -d "stable-diffusion.cpp" ] && [ ! -f "build/bin/sd-cli" ]; then
    echo -e "${GREEN}Cloning repository...${NC}"
    pkg update -y && pkg upgrade -y
    pkg install -y clang cmake git ninja python wget curl
    git clone https://github.com/Juste-Leo2/stable-diffusion.cpp
    cd stable-diffusion.cpp
    git checkout bonsai_dev
elif [ -d "stable-diffusion.cpp" ]; then
    cd stable-diffusion.cpp
fi

if [ ! -f "build/bin/sd-cli" ]; then
    echo -e "${GREEN}Building the project (CPU)...${NC}"
    
    # Termux package installation just in case it wasn't done
    pkg update -y && pkg upgrade -y
    pkg install -y clang cmake git ninja python wget curl

    mkdir -p build && cd build
    
    cmake .. \
      -G Ninja \
      -DCMAKE_BUILD_TYPE=Release

    ninja
    cd ..
fi

echo -e "${GREEN}Checking models...${NC}"
mkdir -p models outputs

if [ ! -f "models/bonsai_image_4b-q1_0.gguf" ]; then
    echo "Downloading Bonsai Model..."
    curl -L "https://huggingface.co/Green-Sky/bonsai-image-binary-4B-GGUF/resolve/main/bonsai_image_4b-q1_0.gguf?download=true" -o models/bonsai_image_4b-q1_0.gguf
fi

if [ ! -f "models/flux2-vae.safetensors" ]; then
    echo "Downloading VAE..."
    curl -L "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors" -o models/flux2-vae.safetensors
fi

if [ ! -f "models/Qwen3-4B-UD-Q3_K_XL.gguf" ]; then
    echo "Downloading Qwen3 LLM..."
    curl -L "https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-UD-Q3_K_XL.gguf?download=true" -o models/Qwen3-4B-UD-Q3_K_XL.gguf
fi

echo -e "${GREEN}Setup complete!${NC}"

# Ask for prompt
read -p "Enter your prompt for image generation (default: 'a cat'): " USER_PROMPT
if [ -z "$USER_PROMPT" ]; then
    USER_PROMPT="a cat"
fi

echo -e "${BLUE}Generating image... This may take a few minutes.${NC}"

./build/bin/sd-cli \
  --diffusion-model models/bonsai_image_4b-q1_0.gguf \
  --vae models/flux2-vae.safetensors \
  --llm models/Qwen3-4B-UD-Q3_K_XL.gguf \
  --cfg-scale 1 \
  --width 512 \
  --height 512 \
  --steps 4 \
  --seed 42 \
  -p "$USER_PROMPT" \
  -o outputs/output.png \
  --mmap

echo -e "${GREEN}Done! Image saved to outputs/output.png${NC}"
