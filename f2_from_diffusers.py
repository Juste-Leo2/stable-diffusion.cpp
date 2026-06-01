#This code was contributed by user Erik Scholz; see: https://huggingface.co/Green-Sky
import torch
import safetensors
import safetensors.torch

tensors = {}

replace_keys = {
    "double_stream_modulation_img.linear.weight": "double_stream_modulation_img.lin.weight",
    "double_stream_modulation_txt.linear.weight": "double_stream_modulation_txt.lin.weight",
    "single_stream_modulation.linear.weight": "single_stream_modulation.lin.weight",
    "context_embedder.weight": "txt_in.weight",
    "norm_out.linear.weight": "final_layer.adaLN_modulation.1.weight",
    "proj_out.weight": "final_layer.linear.weight",
    "x_embedder.weight": "img_in.weight",
    "time_guidance_embed.timestep_embedder.linear_1.weight": "time_in.in_layer.weight",
    "time_guidance_embed.timestep_embedder.linear_2.weight": "time_in.out_layer.weight",
    "single_transformer_blocks.": "single_blocks.",
    "transformer_blocks.": "double_blocks.",
}

single_replace_keys = {
    ".attn.norm_k.weight": ".norm.key_norm.scale",
    ".attn.norm_q.weight": ".norm.query_norm.scale",
    ".attn.to_qkv_mlp_proj.": ".linear1.",
    ".attn.to_out.": ".linear2.",
}

double_replace_keys = {
    ".ff_context.linear_in.": ".txt_mlp.0.",
    ".ff_context.linear_out.": ".txt_mlp.2.",
    ".ff.linear_in.": ".img_mlp.0.",
    ".ff.linear_out.": ".img_mlp.2.",
    ".attn.norm_q.weight": ".img_attn.norm.query_norm.scale",
    ".attn.norm_k.weight": ".img_attn.norm.key_norm.scale",
    ".attn.norm_added_q.weight": ".txt_attn.norm.query_norm.scale",
    ".attn.norm_added_k.weight": ".txt_attn.norm.key_norm.scale",
    ".attn.to_out.0.": ".img_attn.proj.",
    ".attn.to_add_out.": ".txt_attn.proj.",
    ".attn.to_q.": ".img_attn.to_q.", # temp, reshaped later
    ".attn.add_q_proj.": ".txt_attn.to_q.", # temp, reshaped later
}

with safetensors.safe_open("path_to_bonsai_binary/diffusion_pytorch_model.safetensors", framework="pt") as f:
    for k_orig in f.keys():
        k = k_orig
        if k.startswith("model.diffusion_model"):
            k.removeprefix("model.diffusion_model")

        for r, rr in replace_keys.items():
            k = k.replace(r, rr)

        if k.startswith("single_blocks."):
            for r, rr in single_replace_keys.items():
                k = k.replace(r, rr)

        if k.startswith("double_blocks."):
            for r, rr in double_replace_keys.items():
                k = k.replace(r, rr)

        if k.startswith("double_blocks.") and ".to_q" in k:
            q_weight = f.get_tensor(k_orig)
            k_weight = f.get_tensor(k_orig.replace("_q", "_k"))
            v_weight = f.get_tensor(k_orig.replace("_q", "_v"))
            tensors[k.replace("to_q", "qkv")] = torch.cat([q_weight, k_weight, v_weight], dim=0)
            continue

        if not ".attn." in k:
            tensors[k] = f.get_tensor(k_orig)
        else:
            print("left over attn tensor: ", k, " og: ", k_orig)

if "final_layer.adaLN_modulation.1.weight" in tensors:
    w = tensors["final_layer.adaLN_modulation.1.weight"]
    tensors["final_layer.adaLN_modulation.1.weight"] = torch.cat([w[w.shape[0]//2:], w[:w.shape[0]//2]], dim=0)

for k in tensors.keys():
    print(k)

# saving time
print("saving to disk...")
safetensors.torch.save_file(tensors, "bonsai-image-unpacked.safetensors")

print("done.")