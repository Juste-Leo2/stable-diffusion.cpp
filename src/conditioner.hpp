#ifndef __CONDITIONER_HPP__
#define __CONDITIONER_HPP__

#include <cmath>
#include <limits>
#include <optional>

#include "llm.hpp"
#include "tensor_ggml.hpp"

struct SDCondition {
    sd::Tensor<float> c_crossattn;
    sd::Tensor<float> c_vector;
    sd::Tensor<float> c_concat;
    sd::Tensor<int32_t> c_t5_ids;
    sd::Tensor<float> c_t5_weights;
    sd::Tensor<int32_t> c_input_ids;
    sd::Tensor<int32_t> c_position_ids;
    sd::Tensor<int32_t> c_token_types;
    sd::Tensor<int32_t> c_vinput_mask;
    std::vector<std::pair<int, sd::Tensor<float>>> c_image_embeds;
    std::vector<sd::Tensor<float>> c_ref_images;

    std::vector<sd::Tensor<float>> extra_c_crossattns;

    SDCondition() = default;

    SDCondition(sd::Tensor<float> c_crossattn,
                sd::Tensor<float> c_vector,
                sd::Tensor<float> c_concat)
        : c_crossattn(std::move(c_crossattn)), c_vector(std::move(c_vector)), c_concat(std::move(c_concat)) {}

    bool empty() const {
        if (!c_crossattn.empty() || !c_vector.empty() || !c_concat.empty() ||
            !c_t5_ids.empty() || !c_t5_weights.empty() ||
            !c_input_ids.empty() || !c_position_ids.empty() ||
            !c_token_types.empty() || !c_vinput_mask.empty()) {
            return false;
        }

        for (const auto& image_embed : c_image_embeds) {
            if (!image_embed.second.empty()) {
                return false;
            }
        }

        for (const auto& tensor : c_ref_images) {
            if (!tensor.empty()) {
                return false;
            }
        }

        for (const auto& tensor : extra_c_crossattns) {
            if (!tensor.empty()) {
                return false;
            }
        }

        return true;
    }
};

static inline sd::Tensor<float> apply_token_weights(sd::Tensor<float> hidden_states,
                                                    const std::vector<float>& weights) {
    if (hidden_states.empty()) {
        return hidden_states;
    }

    bool all_one = true;
    for (float weight : weights) {
        if (weight != 1.0f) {
            all_one = false;
            break;
        }
    }
    if (all_one) {
        return hidden_states;
    }

    if (hidden_states.dim() == 1) {
        hidden_states.unsqueeze_(1);
    }

    GGML_ASSERT(static_cast<size_t>(hidden_states.shape()[1]) == weights.size());

    float original_mean = hidden_states.mean();
    auto chunk_weights  = sd::Tensor<float>::from_vector(weights);
    chunk_weights.reshape_({1, static_cast<int64_t>(weights.size())});
    hidden_states *= chunk_weights;
    float new_mean = hidden_states.mean();
    if (std::isfinite(original_mean) && std::isfinite(new_mean) && new_mean != 0.0f) {
        hidden_states *= (original_mean / new_mean);
    }

    return hidden_states;
}

struct ConditionerParams {
    std::string text;
    int clip_skip                                    = -1;
    int width                                        = -1;
    int height                                       = -1;
    bool zero_out_masked                             = false;
    int num_input_imgs                               = 0;        // for photomaker
    const std::vector<sd::Tensor<float>>* ref_images = nullptr;  // for qwen image edit
};

struct Conditioner {
    virtual ~Conditioner() = default;

public:
    virtual SDCondition get_learned_condition(int n_threads,
                                              const ConditionerParams& conditioner_params) = 0;
    virtual void alloc_params_buffer()                                                     = 0;
    virtual void free_params_buffer()                                                      = 0;
    virtual void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors)           = 0;
    virtual size_t get_params_buffer_size()                                                = 0;
    virtual void set_max_graph_vram_bytes(size_t max_vram_bytes) {}
    virtual void set_flash_attention_enabled(bool enabled) = 0;
    virtual void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) {}
    virtual std::tuple<SDCondition, std::vector<bool>> get_learned_condition_with_trigger(int n_threads,
                                                                                          const ConditionerParams& conditioner_params) {
        GGML_ABORT("Not implemented yet!");
    }
    virtual std::string remove_trigger_from_prompt(const std::string& prompt) {
        GGML_ABORT("Not implemented yet!");
    }
};

struct LLMEmbedder : public Conditioner {
    SDVersion version;
    std::shared_ptr<BPETokenizer> tokenizer;
    std::shared_ptr<LLM::LLMRunner> llm;

    LLMEmbedder(ggml_backend_t backend,
                ggml_backend_t params_backend,
                const String2TensorStorage& tensor_storage_map = {},
                SDVersion version                              = VERSION_QWEN_IMAGE,
                const std::string prefix                       = "",
                bool enable_vision                             = false)
        : version(version) {
        LLM::LLMArch arch = LLM::LLMArch::QWEN2_5_VL;
        if (version == VERSION_FLUX2) {
            arch = LLM::LLMArch::MISTRAL_SMALL_3_2;
        } else if (sd_version_is_ernie_image(version)) {
            arch = LLM::LLMArch::MINISTRAL_3_3B;
        } else if (sd_version_is_lens(version)) {
            arch = LLM::LLMArch::GPT_OSS_20B;
        } else if (sd_version_is_z_image(version) || version == VERSION_OVIS_IMAGE || version == VERSION_FLUX2_KLEIN) {
            arch = LLM::LLMArch::QWEN3;
        }
        {
            tokenizer = std::make_shared<Qwen2Tokenizer>();
        }
        llm = std::make_shared<LLM::LLMRunner>(arch,
                                               backend,
                                               params_backend,
                                               tensor_storage_map,
                                               "text_encoders.llm",
                                               enable_vision);
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        llm->get_param_tensors(tensors, "text_encoders.llm");
    }

    void alloc_params_buffer() override {
        llm->alloc_params_buffer();
    }

    void free_params_buffer() override {
        llm->free_params_buffer();
    }

    size_t get_params_buffer_size() override {
        size_t buffer_size = 0;
        buffer_size += llm->get_params_buffer_size();
        return buffer_size;
    }

    void set_max_graph_vram_bytes(size_t max_vram_bytes) override {
        llm->set_max_graph_vram_bytes(max_vram_bytes);
    }

    void set_flash_attention_enabled(bool enabled) override {
        llm->set_flash_attention_enabled(enabled);
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        if (llm) {
            llm->set_weight_adapter(adapter);
        }
    }

    std::tuple<std::vector<int>, std::vector<float>, std::vector<float>> tokenize(std::string text,
                                                                                  const std::pair<int, int>& attn_range,
                                                                                  size_t min_length = 0,
                                                                                  size_t max_length = 100000000,
                                                                                  bool spell_quotes = false) {
        std::vector<std::pair<std::string, float>> parsed_attention;
        if (attn_range.first >= 0 && attn_range.second > 0) {
            if (attn_range.first > 0) {
                parsed_attention.emplace_back(text.substr(0, attn_range.first), 1.f);
            }
            if (attn_range.second - attn_range.first > 0) {
                auto new_parsed_attention = parse_prompt_attention(text.substr(attn_range.first, attn_range.second - attn_range.first));
                if (spell_quotes) {
                    new_parsed_attention = split_quotation_attention(new_parsed_attention);
                }
                parsed_attention.insert(parsed_attention.end(),
                                        new_parsed_attention.begin(),
                                        new_parsed_attention.end());
            }
            if (attn_range.second < text.size()) {
                parsed_attention.emplace_back(text.substr(attn_range.second), 1.f);
            }
        } else {
            parsed_attention.emplace_back(text, 1.f);
        }

        {
            std::stringstream ss;
            ss << "[";
            for (const auto& item : parsed_attention) {
                ss << "['" << item.first << "', " << item.second << "], ";
            }
            ss << "]";
            LOG_DEBUG("parse '%s' to %s", text.c_str(), ss.str().c_str());
        }

        std::vector<int> tokens;
        std::vector<float> weights;
        for (const auto& item : parsed_attention) {
            const std::string& curr_text = item.first;
            float curr_weight            = item.second;
            std::vector<int> curr_tokens = tokenizer->encode(curr_text, nullptr);
            tokens.insert(tokens.end(), curr_tokens.begin(), curr_tokens.end());
            weights.insert(weights.end(), curr_tokens.size(), curr_weight);
        }

        std::vector<float> mask;
        tokenizer->pad_tokens(tokens, &weights, &mask, min_length, max_length);

        // for (int i = 0; i < tokens.size(); i++) {
        //     std::cout << tokens[i] << ":" << weights[i] << ", " << i << std::endl;
        // }
        // std::cout << std::endl;

        return {tokens, weights, mask};
    }

    sd::Tensor<float> encode_prompt(int n_threads,
                                    const std::string prompt,
                                    const std::pair<int, int>& prompt_attn_range,
                                    int min_length,
                                    int hidden_states_min_length,
                                    const std::vector<std::pair<int, sd::Tensor<float>>>& image_embeds,
                                    const std::set<int>& out_layers,
                                    int prompt_template_encode_start_idx,
                                    bool spell_quotes = false,
                                    int max_length    = 100000000) {
        auto tokens_weights_mask = tokenize(prompt, prompt_attn_range, min_length, max_length, spell_quotes);
        auto& tokens             = std::get<0>(tokens_weights_mask);
        auto& weights            = std::get<1>(tokens_weights_mask);
        auto& mask               = std::get<2>(tokens_weights_mask);

        sd::Tensor<int32_t> input_ids({static_cast<int64_t>(tokens.size())}, tokens);
        sd::Tensor<float> attention_mask;
        if (!mask.empty()) {
            attention_mask = sd::Tensor<float>({static_cast<int64_t>(mask.size()), static_cast<int64_t>(mask.size())});
            for (size_t i1 = 0; i1 < mask.size(); ++i1) {
                for (size_t i0 = 0; i0 < mask.size(); ++i0) {
                    float value = 0.0f;
                    if (mask[i0] == 0.0f || i0 > i1) {
                        value = -INFINITY;
                    }
                    attention_mask[static_cast<int64_t>(i0 + mask.size() * i1)] = value;
                }
            }
        }

        auto hidden_states = llm->compute(n_threads,
                                          input_ids,
                                          attention_mask,
                                          image_embeds,
                                          out_layers);
        GGML_ASSERT(!hidden_states.empty());
        hidden_states = apply_token_weights(std::move(hidden_states), weights);
        GGML_ASSERT(hidden_states.shape()[1] > prompt_template_encode_start_idx);

        int64_t zero_pad_len = 0;
        if (hidden_states_min_length > 0) {
            if (hidden_states.shape()[1] - prompt_template_encode_start_idx < hidden_states_min_length) {
                zero_pad_len = hidden_states_min_length - hidden_states.shape()[1] + prompt_template_encode_start_idx;
            }
        }

        sd::Tensor<float> new_hidden_states = sd::ops::slice(hidden_states,
                                                             1,
                                                             prompt_template_encode_start_idx,
                                                             hidden_states.shape()[1]);
        if (zero_pad_len > 0) {
            auto pad_shape    = new_hidden_states.shape();
            pad_shape[1]      = zero_pad_len;
            new_hidden_states = sd::ops::concat(new_hidden_states,
                                                sd::Tensor<float>::zeros(std::move(pad_shape)),
                                                1);
        }

        return new_hidden_states;
    }

    SDCondition get_learned_condition(int n_threads,
                                      const ConditionerParams& conditioner_params) override {
        std::string prompt;
        std::pair<int, int> prompt_attn_range;
        std::vector<std::string> extra_prompts;
        std::vector<std::pair<int, int>> extra_prompts_attn_range;
        std::vector<std::pair<int, sd::Tensor<float>>> image_embeds;
        int prompt_template_encode_start_idx = 34;
        int min_length                       = 0;  // pad tokens
        int max_length                       = 100000000;
        int hidden_states_min_length         = 0;  // zero pad hidden_states
        bool spell_quotes                    = false;
        std::set<int> out_layers;

        int64_t t0 = ggml_time_ms();

        if (sd_version_is_qwen_image(version)) {
            if (llm->enable_vision && conditioner_params.ref_images != nullptr && !conditioner_params.ref_images->empty()) {
                LOG_INFO("QwenImageEditPlusPipeline");
                prompt_template_encode_start_idx = 64;
                int image_embed_idx              = 64 + 6;

                int min_pixels          = 384 * 384;
                int max_pixels          = 560 * 560;
                std::string placeholder = "<|image_pad|>";
                std::string img_prompt;

                for (int i = 0; i < conditioner_params.ref_images->size(); i++) {
                    const auto& image = (*conditioner_params.ref_images)[i];
                    double factor     = llm->params.vision.patch_size * llm->params.vision.spatial_merge_size;
                    int height        = static_cast<int>(image.shape()[1]);
                    int width         = static_cast<int>(image.shape()[0]);
                    int h_bar         = static_cast<int>(std::round(height / factor) * factor);
                    int w_bar         = static_cast<int>(std::round(width / factor) * factor);

                    if (static_cast<double>(h_bar) * w_bar > max_pixels) {
                        double beta = std::sqrt((height * width) / static_cast<double>(max_pixels));
                        h_bar       = std::max(static_cast<int>(factor),
                                               static_cast<int>(std::floor(height / beta / factor)) * static_cast<int>(factor));
                        w_bar       = std::max(static_cast<int>(factor),
                                               static_cast<int>(std::floor(width / beta / factor)) * static_cast<int>(factor));
                    } else if (static_cast<double>(h_bar) * w_bar < min_pixels) {
                        double beta = std::sqrt(static_cast<double>(min_pixels) / (height * width));
                        h_bar       = static_cast<int>(std::ceil(height * beta / factor)) * static_cast<int>(factor);
                        w_bar       = static_cast<int>(std::ceil(width * beta / factor)) * static_cast<int>(factor);
                    }

                    LOG_DEBUG("resize conditioner ref image %d from %dx%d to %dx%d", i, height, width, h_bar, w_bar);

                    auto resized_image = clip_preprocess(image, w_bar, h_bar);

                    auto image_embed = llm->encode_image(n_threads, resized_image);
                    GGML_ASSERT(!image_embed.empty());
                    image_embeds.emplace_back(image_embed_idx, image_embed);
                    image_embed_idx += 1 + static_cast<int>(image_embed.shape()[1]) + 6;

                    img_prompt += "Picture " + std::to_string(i + 1) + ": <|vision_start|>";  // [24669, 220, index, 25, 220, 151652]
                    int64_t num_image_tokens = image_embed.shape()[1];
                    img_prompt.reserve(num_image_tokens * placeholder.size());
                    for (int j = 0; j < num_image_tokens; j++) {
                        img_prompt += placeholder;
                    }
                    img_prompt += "<|vision_end|>";
                }

                prompt = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n";
                prompt += img_prompt;

                prompt_attn_range.first = static_cast<int>(prompt.size());
                prompt += conditioner_params.text;
                prompt_attn_range.second = static_cast<int>(prompt.size());

                prompt += "<|im_end|>\n<|im_start|>assistant\n";
            } else {
                prompt_template_encode_start_idx = 34;

                prompt = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n";

                prompt_attn_range.first = static_cast<int>(prompt.size());
                prompt += conditioner_params.text;
                prompt_attn_range.second = static_cast<int>(prompt.size());

                prompt += "<|im_end|>\n<|im_start|>assistant\n";
            }
        } else if (sd_version_is_longcat(version)) {
            spell_quotes = true;

            if (llm->enable_vision && conditioner_params.ref_images != nullptr && !conditioner_params.ref_images->empty()) {
                LOG_INFO("LongCatEditPipeline");
                prompt_template_encode_start_idx = 67;
                min_length                       = 512 + prompt_template_encode_start_idx;
                int image_embed_idx              = 36 + 6;

                int min_pixels          = 384 * 384;
                int max_pixels          = 560 * 560;
                std::string placeholder = "<|image_pad|>";
                std::string img_prompt;

                for (int i = 0; i < conditioner_params.ref_images->size(); i++) {
                    const auto& image = (*conditioner_params.ref_images)[i];
                    double factor     = llm->params.vision.patch_size * llm->params.vision.spatial_merge_size;
                    int height        = static_cast<int>(image.shape()[1]);
                    int width         = static_cast<int>(image.shape()[0]);
                    int h_bar         = static_cast<int>(std::round(height / factor) * factor);
                    int w_bar         = static_cast<int>(std::round(width / factor) * factor);

                    if (static_cast<double>(h_bar) * w_bar > max_pixels) {
                        double beta = std::sqrt((height * width) / static_cast<double>(max_pixels));
                        h_bar       = std::max(static_cast<int>(factor),
                                               static_cast<int>(std::floor(height / beta / factor)) * static_cast<int>(factor));
                        w_bar       = std::max(static_cast<int>(factor),
                                               static_cast<int>(std::floor(width / beta / factor)) * static_cast<int>(factor));
                    } else if (static_cast<double>(h_bar) * w_bar < min_pixels) {
                        double beta = std::sqrt(static_cast<double>(min_pixels) / (height * width));
                        h_bar       = static_cast<int>(std::ceil(height * beta / factor)) * static_cast<int>(factor);
                        w_bar       = static_cast<int>(std::ceil(width * beta / factor)) * static_cast<int>(factor);
                    }

                    LOG_DEBUG("resize conditioner ref image %d from %dx%d to %dx%d", i, height, width, h_bar, w_bar);

                    auto resized_image = clip_preprocess(image, w_bar, h_bar);
                    auto image_embed   = llm->encode_image(n_threads, resized_image);
                    GGML_ASSERT(!image_embed.empty());
                    image_embeds.emplace_back(image_embed_idx, image_embed);
                    image_embed_idx += 1 + static_cast<int>(image_embed.shape()[1]) + 6;

                    img_prompt += "<|vision_start|>";
                    int64_t num_image_tokens = image_embed.shape()[1];
                    img_prompt.reserve(num_image_tokens * placeholder.size());
                    for (int j = 0; j < num_image_tokens; j++) {
                        img_prompt += placeholder;
                    }
                    img_prompt += "<|vision_end|>";
                }

                prompt = "<|im_start|>system\nAs an image editing expert, first analyze the content and attributes of the input image(s). Then, based on the user's editing instructions, clearly and precisely determine how to modify the given image(s), ensuring that only the specified parts are altered and all other aspects remain consistent with the original(s).<|im_end|>\n<|im_start|>user\n";
                prompt += img_prompt;
            } else {
                prompt_template_encode_start_idx = 36;
                min_length                       = 512 + prompt_template_encode_start_idx;

                prompt = "<|im_start|>system\nAs an image captioning expert, generate a descriptive text prompt based on an image content, suitable for input to a text-to-image model.<|im_end|>\n<|im_start|>user\n";
            }

            prompt_attn_range.first = static_cast<int>(prompt.size());
            prompt += conditioner_params.text;
            prompt_attn_range.second = static_cast<int>(prompt.size());

            prompt += "<|im_end|>\n<|im_start|>assistant\n";
        } else if (version == VERSION_FLUX2) {
            prompt_template_encode_start_idx = 0;
            hidden_states_min_length         = 512;
            out_layers                       = {10, 20, 30};

            prompt = "[SYSTEM_PROMPT]You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object\nattribution and actions without speculation.[/SYSTEM_PROMPT][INST]";

            prompt_attn_range.first = static_cast<int>(prompt.size());
            prompt += conditioner_params.text;
            prompt_attn_range.second = static_cast<int>(prompt.size());

            prompt += "[/INST]";
        } else if (sd_version_is_ernie_image(version)) {
            prompt_template_encode_start_idx = 0;
            out_layers                       = {25};  // -2

            prompt_attn_range.first = 0;
            prompt += conditioner_params.text;
            prompt_attn_range.second = static_cast<int>(prompt.size());
        } else if (sd_version_is_lens(version)) {
            prompt_template_encode_start_idx = 97;
            min_length                       = 0;
            max_length                       = 512;
            out_layers                       = {6, 12, 18, 24};

            prompt =
                "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
                "Knowledge cutoff: 2024-06\n"
                "Current date: 2026-05-26\n"  // fix for current date
                "\n"
                "Reasoning: medium\n"
                "\n"
                "# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>developer<|message|># Instructions\n"
                "\n"
                "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background.\n"
                "\n"
                "<|end|><|start|>user<|message|>";

            prompt_attn_range.first = static_cast<int>(prompt.size());
            prompt += conditioner_params.text;
            prompt_attn_range.second = static_cast<int>(prompt.size());

            prompt += "<|end|><|start|>assistant<|channel|>analysis<|message|>Need to generate one image according to the description.<|end|><|start|>assistant<|channel|>final<|message|>";
        } else if (sd_version_is_z_image(version)) {
            prompt_template_encode_start_idx = 0;
            out_layers                       = {35};  // -2

            if (conditioner_params.ref_images != nullptr && !conditioner_params.ref_images->empty()) {
                LOG_INFO("ZImageOmniPipeline");
                prompt = "<|im_start|>user\n<|vision_start|>";
                for (int i = 0; i < conditioner_params.ref_images->size() - 1; i++) {
                    extra_prompts.push_back("<|vision_end|><|vision_start|>");
                }
                extra_prompts.push_back("<|vision_end|>" + conditioner_params.text + "<|im_end|>\n<|im_start|>assistant\n<|vision_start|>");
                extra_prompts.push_back("<|vision_end|><|im_end|>");
            } else {
                prompt = "<|im_start|>user\n";

                prompt_attn_range.first = static_cast<int>(prompt.size());
                prompt += conditioner_params.text;
                prompt_attn_range.second = static_cast<int>(prompt.size());

                prompt += "<|im_end|>\n<|im_start|>assistant\n";
            }
        } else if (version == VERSION_FLUX2_KLEIN) {
            prompt_template_encode_start_idx = 0;
            min_length                       = 512;
            out_layers                       = {9, 18, 27};

            prompt = "<|im_start|>user\n";

            prompt_attn_range.first = static_cast<int>(prompt.size());
            prompt += conditioner_params.text;
            prompt_attn_range.second = static_cast<int>(prompt.size());

            prompt += "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        } else if (version == VERSION_OVIS_IMAGE) {
            prompt_template_encode_start_idx = 28;
            min_length                       = prompt_template_encode_start_idx + 256;

            prompt = "<|im_start|>user\nDescribe the image by detailing the color, quantity, text, shape, size, texture, spatial relationships of the objects and background:";

            prompt_attn_range.first = static_cast<int>(prompt.size());
            prompt += " " + conditioner_params.text;
            prompt_attn_range.second = static_cast<int>(prompt.size());

            prompt += "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        } else {
            GGML_ABORT("unknown version %d", version);
        }

        auto hidden_states = encode_prompt(n_threads,
                                           prompt,
                                           prompt_attn_range,
                                           min_length,
                                           hidden_states_min_length,
                                           image_embeds,
                                           out_layers,
                                           prompt_template_encode_start_idx,
                                           spell_quotes,
                                           max_length);
        std::vector<sd::Tensor<float>> extra_hidden_states_vec;
        for (int i = 0; i < extra_prompts.size(); i++) {
            auto extra_hidden_states = encode_prompt(n_threads,
                                                     extra_prompts[i],
                                                     extra_prompts_attn_range[i],
                                                     min_length,
                                                     hidden_states_min_length,
                                                     image_embeds,
                                                     out_layers,
                                                     prompt_template_encode_start_idx,
                                                     spell_quotes,
                                                     max_length);
            extra_hidden_states_vec.push_back(std::move(extra_hidden_states));
        }

        int64_t t1 = ggml_time_ms();
        LOG_DEBUG("computing condition graph completed, taking %" PRId64 " ms", t1 - t0);
        SDCondition result;
        result.c_crossattn        = std::move(hidden_states);
        result.extra_c_crossattns = std::move(extra_hidden_states_vec);
        return result;
    }
};

#endif
