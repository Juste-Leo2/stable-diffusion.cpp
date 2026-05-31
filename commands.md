# Commands

## Download model

```sh
hf download prism-ml/bonsai-image-binary-4B-unpacked --local-dir models/bonsai-image-binary-4B-unpacked
```

## Merge sharded text encoder (required for both C++ and Python)

```sh
uv run python3 -c "
from safetensors.torch import save_file
from safetensors import safe_open
import os

d = 'models/bonsai-image-binary-4B-unpacked/text_encoder'
tensors = {}
for shard in ['model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors']:
    with safe_open(os.path.join(d, shard), framework='pt') as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
save_file(tensors, os.path.join(d, 'model.safetensors'))
print('merged:', os.path.getsize(os.path.join(d, 'model.safetensors')) / 1e9, 'GB')
"
```

## Build C++ with CUDA

```sh
git submodule init && git submodule update
cmake -B build -DCMAKE_BUILD_TYPE=Release -DSD_CUDA=ON
cmake --build build -j
```

## C++ inference

```sh
mkdir -p outputs
./build/bin/sd-cli \
  --cfg-scale 1 --width 512 --height 512 --steps 4 --seed 42 \
  -p "a cat sitting on a window sill" \
  -o outputs/cat_cpp.png \
  --diffusion-model models/bonsai-image-binary-4B-unpacked/transformer/diffusion_pytorch_model.safetensors \
  --vae models/bonsai-image-binary-4B-unpacked/vae/diffusion_pytorch_model.safetensors \
  --llm models/bonsai-image-binary-4B-unpacked/text_encoder/model.safetensors
```

## Python dependencies

```sh
uv pip install torch diffusers transformers pillow accelerate
```

## Python inference

```sh
mkdir -p outputs
uv run python3 image-studio/backend_gpu/scripts/inference_bf16.py \
  --prompt "a cat sitting on a window sill" \
  --output outputs/cat_python.png
```
