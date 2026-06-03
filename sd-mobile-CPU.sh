#!/bin/bash

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== Stable-Diffusion.cpp Termux Setup (CPU) ===${NC}"

[ ! -d "stable-diffusion.cpp" ] && [ ! -f "build/bin/sd-cli" ] && echo -e "${GREEN}Cloning repository...${NC}" && pkg update -y && pkg upgrade -y && pkg install -y clang cmake git ninja python wget curl && git clone https://github.com/Juste-Leo2/stable-diffusion.cpp && cd stable-diffusion.cpp && git checkout bonsai_dev
[ -d "stable-diffusion.cpp" ] && cd stable-diffusion.cpp

[ ! -f "build/bin/sd-cli" ] && echo -e "${GREEN}Building the project (CPU)...${NC}" && pkg update -y && pkg upgrade -y && pkg install -y clang cmake git ninja python wget curl && mkdir -p build && cd build && cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release && ninja -j 4 && cd ..

echo -e "${GREEN}Checking models...${NC}"
mkdir -p models outputs

[ ! -f "models/bonsai_image_4b-q1_0.gguf" ] && echo "Downloading Bonsai Model..." && curl -L "https://huggingface.co/Green-Sky/bonsai-image-binary-4B-GGUF/resolve/main/bonsai_image_4b-q1_0.gguf?download=true" -o models/bonsai_image_4b-q1_0.gguf

[ ! -f "models/flux2-vae.safetensors" ] && echo "Downloading VAE..." && curl -L "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors" -o models/flux2-vae.safetensors

[ ! -f "models/Qwen3-4B-UD-Q3_K_XL.gguf" ] && echo "Downloading Qwen3 LLM..." && curl -L "https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-UD-Q3_K_XL.gguf?download=true" -o models/Qwen3-4B-UD-Q3_K_XL.gguf

echo -e "${GREEN}Setup complete!${NC}"

echo -e "${BLUE}Generating image... This may take a few minutes.${NC}"

./build/bin/sd-cli --diffusion-model models/bonsai_image_4b-q1_0.gguf --vae models/flux2-vae.safetensors --llm models/Qwen3-4B-UD-Q3_K_XL.gguf --cfg-scale 1 --width 512 --height 512 --steps 6 --seed 42 -p "A cat is strolling through a park; the sun is shining; the cat is alone; in the background, there is a magnificent mountain landscape" -o outputs/output.png --mmap --fa

echo -e "${GREEN}Done! Image saved to outputs/output.png${NC}"
