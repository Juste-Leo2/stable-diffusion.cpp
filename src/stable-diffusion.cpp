#include "ggml_extend.hpp"
#include "ggml_graph_cut.h"

#include "model.h"
#include "rng.hpp"
#include "rng_mt19937.hpp"
#include "rng_philox.hpp"
#include "stable-diffusion.h"
#include "util.h"

#include "auto_encoder_kl.hpp"
#include "conditioner.hpp"
#include "denoiser.hpp"
#include "diffusion_model.hpp"
#include "flux.hpp"
#include "guidance.h"
#include "latent-preview.h"
#include "lora.hpp"
#include "tae.hpp"
#include "vae.hpp"
#include "name_conversion.h"

const char* model_version_to_str[] = {
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "Flux.2",
    "Flux.2 klein",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
};

const char* sampling_methods_str[] = {
    "Euler",
    "Euler A",
    "Heun",
    "DPM2",
    "DPM++ (2s)",
    "DPM++ (2M)",
    "modified DPM++ (2M)",
    "iPNDM",
    "iPNDM_v",
    "LCM",
    "DDIM \"trailing\"",
    "TCD",
    "Res Multistep",
    "Res 2s",
    "ER-SDE",
    "Euler CFG++",
    "Euler A CFG++",
    "Euler GE",
};

/*================================================== Helper Functions ================================================*/

void calculate_alphas_cumprod(float* alphas_cumprod,
                              float linear_start = 0.00085f,
                              float linear_end   = 0.0120f,
                              int timesteps      = TIMESTEPS) {
    float ls_sqrt = sqrtf(linear_start);
    float le_sqrt = sqrtf(linear_end);
    float amount  = le_sqrt - ls_sqrt;
    float product = 1.0f;
    for (int i = 0; i < timesteps; i++) {
        float beta = ls_sqrt + amount * ((float)i / (timesteps - 1));
        product *= 1.0f - powf(beta, 2.0f);
        alphas_cumprod[i] = product;
    }
}

static float get_cache_reuse_threshold(const sd_cache_params_t& params) {
    float reuse_threshold = params.reuse_threshold;
    if (reuse_threshold == INFINITY) {
        if (params.mode == SD_CACHE_EASYCACHE) {
            reuse_threshold = 0.2f;
        } else if (params.mode == SD_CACHE_UCACHE) {
            reuse_threshold = 1.0f;
        }
    }
    return std::max(0.0f, reuse_threshold);
}

/*=============================================== StableDiffusionGGML ================================================*/

class StableDiffusionGGML {
public:
    std::vector<MmapTensorStore> mmap_tensor_store;
    SDBackendManager backend_manager;

    SDVersion version;
    bool vae_decode_only         = false;
    bool external_vae_is_invalid = false;
    bool free_params_immediately = false;

    bool circular_x = false;
    bool circular_y = false;

    std::shared_ptr<RNG> rng         = std::make_shared<PhiloxRNG>();
    std::shared_ptr<RNG> sampler_rng = nullptr;
    int n_threads                    = -1;
    float default_flow_shift         = INFINITY;

    std::shared_ptr<Conditioner> cond_stage_model;
    std::shared_ptr<DiffusionModelRunner> diffusion_model;
    std::shared_ptr<VAE> first_stage_model;
    std::shared_ptr<VAE> preview_vae;
    std::shared_ptr<LoraModel> pmid_lora;
    std::vector<std::shared_ptr<LoraModel>> cond_stage_lora_models;
    std::vector<std::shared_ptr<LoraModel>> diffusion_lora_models;
    std::vector<std::shared_ptr<LoraModel>> first_stage_lora_models;
    bool apply_lora_immediately = false;

    std::string taesd_path;
    sd_tiling_params_t vae_tiling_params = {false, false, 0, 0, 0.5f, 0, 0, nullptr};
    bool offload_params_to_cpu           = false;
    float max_vram                       = 0.f;
    bool use_pmid                        = false;
    std::string backend_spec;
    std::string params_backend_spec;

    bool is_using_v_parameterization     = false;
    bool is_using_edm_v_parameterization = false;

    std::map<std::string, ggml_tensor*> tensors;

    // lora_name => multiplier
    std::unordered_map<std::string, float> curr_lora_state;

    std::shared_ptr<Denoiser> denoiser = std::make_shared<CompVisDenoiser>();

    StableDiffusionGGML() = default;

    ~StableDiffusionGGML() = default;

    ggml_backend_t backend_for(SDBackendModule module) {
        ggml_backend_t module_backend = backend_manager.runtime_backend(module);
        if (module_backend == nullptr) {
            LOG_ERROR("failed to initialize %s backend", sd_backend_module_name(module));
        }
        return module_backend;
    }

    ggml_backend_t params_backend_for(SDBackendModule module) {
        ggml_backend_t module_backend = backend_manager.params_backend(module);
        if (module_backend == nullptr) {
            LOG_ERROR("failed to initialize %s params backend", sd_backend_module_name(module));
        }
        return module_backend;
    }

    bool ensure_backend_pair(SDBackendModule module) {
        if (backend_for(module) == nullptr) {
            return false;
        }
        return params_backend_for(module) != nullptr;
    }

    bool init_backend(const sd_ctx_params_t* sd_ctx_params) {
        std::string error;
        if (!backend_manager.init(sd_ctx_params->backend,
                                  sd_ctx_params->params_backend,
                                  sd_ctx_params->offload_params_to_cpu,
                                  sd_ctx_params->keep_clip_on_cpu,
                                  sd_ctx_params->keep_vae_on_cpu,
                                  sd_ctx_params->keep_control_net_on_cpu,
                                  &error)) {
            LOG_ERROR("backend config failed: %s", error.c_str());
            return false;
        }
        return ensure_backend_pair(SDBackendModule::DIFFUSION);
    }

    std::shared_ptr<RNG> get_rng(rng_type_t rng_type) {
        if (rng_type == STD_DEFAULT_RNG) {
            return std::make_shared<STDDefaultRNG>();
        } else if (rng_type == CPU_RNG) {
            return std::make_shared<MT19937RNG>();
        } else {  // default: CUDA_RNG
            return std::make_shared<PhiloxRNG>();
        }
    }

    bool init(const sd_ctx_params_t* sd_ctx_params) {
        n_threads               = sd_ctx_params->n_threads;
        vae_decode_only         = sd_ctx_params->vae_decode_only;
        free_params_immediately = sd_ctx_params->free_params_immediately;
        offload_params_to_cpu   = sd_ctx_params->offload_params_to_cpu;
        max_vram                = sd_ctx_params->max_vram;
        backend_spec            = SAFE_STR(sd_ctx_params->backend);
        params_backend_spec     = SAFE_STR(sd_ctx_params->params_backend);

        bool use_tae = false;

        rng = get_rng(sd_ctx_params->rng_type);
        if (sd_ctx_params->sampler_rng_type != RNG_TYPE_COUNT && sd_ctx_params->sampler_rng_type != sd_ctx_params->rng_type) {
            sampler_rng = get_rng(sd_ctx_params->sampler_rng_type);
        } else {
            sampler_rng = rng;
        }

        ggml_log_set(ggml_log_callback_default, nullptr);

        if (!init_backend(sd_ctx_params)) {
            return false;
        }
        max_vram = sd::ggml_graph_cut::resolve_max_vram_gib(max_vram, backend_for(SDBackendModule::DIFFUSION));

        ModelLoader model_loader;

        if (strlen(SAFE_STR(sd_ctx_params->model_path)) > 0) {
            LOG_INFO("loading model from '%s'", sd_ctx_params->model_path);
            if (!model_loader.init_from_file(sd_ctx_params->model_path)) {
                LOG_ERROR("init model loader from file failed: '%s'", sd_ctx_params->model_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->diffusion_model_path)) > 0) {
            LOG_INFO("loading diffusion model from '%s'", sd_ctx_params->diffusion_model_path);
            if (!model_loader.init_from_file(sd_ctx_params->diffusion_model_path, "model.diffusion_model.")) {
                LOG_WARN("loading diffusion model from '%s' failed", sd_ctx_params->diffusion_model_path);
            }
        }

        bool is_unet = sd_version_is_unet(model_loader.get_sd_version());

        if (strlen(SAFE_STR(sd_ctx_params->clip_l_path)) > 0) {
            LOG_INFO("loading clip_l from '%s'", sd_ctx_params->clip_l_path);
            std::string prefix = is_unet ? "cond_stage_model.transformer." : "text_encoders.clip_l.transformer.";
            if (!model_loader.init_from_file(sd_ctx_params->clip_l_path, prefix)) {
                LOG_WARN("loading clip_l from '%s' failed", sd_ctx_params->clip_l_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->clip_g_path)) > 0) {
            LOG_INFO("loading clip_g from '%s'", sd_ctx_params->clip_g_path);
            std::string prefix = is_unet ? "cond_stage_model.1.transformer." : "text_encoders.clip_g.transformer.";
            if (!model_loader.init_from_file(sd_ctx_params->clip_g_path, prefix)) {
                LOG_WARN("loading clip_g from '%s' failed", sd_ctx_params->clip_g_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->t5xxl_path)) > 0) {
            LOG_INFO("loading t5xxl from '%s'", sd_ctx_params->t5xxl_path);
            if (!model_loader.init_from_file(sd_ctx_params->t5xxl_path, "text_encoders.t5xxl.transformer.")) {
                LOG_WARN("loading t5xxl from '%s' failed", sd_ctx_params->t5xxl_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->llm_path)) > 0) {
            LOG_INFO("loading llm from '%s'", sd_ctx_params->llm_path);
            if (!model_loader.init_from_file(sd_ctx_params->llm_path, "text_encoders.llm.")) {
                LOG_WARN("loading llm from '%s' failed", sd_ctx_params->llm_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->llm_vision_path)) > 0) {
            LOG_INFO("loading llm vision from '%s'", sd_ctx_params->llm_vision_path);
            if (!model_loader.init_from_file(sd_ctx_params->llm_vision_path, "text_encoders.llm.visual.")) {
                LOG_WARN("loading llm vision from '%s' failed", sd_ctx_params->llm_vision_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->vae_path)) > 0) {
            LOG_INFO("loading vae from '%s'", sd_ctx_params->vae_path);
            if (!model_loader.init_from_file(sd_ctx_params->vae_path, "vae.")) {
                LOG_WARN("loading vae from '%s' failed", sd_ctx_params->vae_path);
                external_vae_is_invalid = true;
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->taesd_path)) > 0) {
            LOG_INFO("loading tae from '%s'", sd_ctx_params->taesd_path);
            if (!model_loader.init_from_file(sd_ctx_params->taesd_path, "tae.")) {
                LOG_WARN("loading tae from '%s' failed", sd_ctx_params->taesd_path);
            } else {
                use_tae = true;
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->embeddings_connectors_path)) > 0) {
            LOG_INFO("loading embeddings connectors from '%s'", sd_ctx_params->embeddings_connectors_path);
            if (!model_loader.init_from_file(sd_ctx_params->embeddings_connectors_path)) {
                LOG_WARN("loading embeddings connectors from '%s' failed", sd_ctx_params->embeddings_connectors_path);
            }
        }

        model_loader.convert_tensors_name();

        version = model_loader.get_sd_version();
        if (version == VERSION_COUNT) {
            LOG_ERROR("get sd version from file failed: '%s'", SAFE_STR(sd_ctx_params->model_path));
            return false;
        }

        auto& tensor_storage_map = model_loader.get_tensor_storage_map();

        LOG_INFO("Version: %s ", model_version_to_str[version]);
        ggml_type wtype               = (int)sd_ctx_params->wtype < std::min<int>(SD_TYPE_COUNT, GGML_TYPE_COUNT)
                                            ? (ggml_type)sd_ctx_params->wtype
                                            : GGML_TYPE_COUNT;
        std::string tensor_type_rules = SAFE_STR(sd_ctx_params->tensor_type_rules);
        if (wtype != GGML_TYPE_COUNT || tensor_type_rules.size() > 0) {
            model_loader.set_wtype_override(wtype, tensor_type_rules);
        }

        std::map<ggml_type, uint32_t> wtype_stat                 = model_loader.get_wtype_stat();
        std::map<ggml_type, uint32_t> conditioner_wtype_stat     = model_loader.get_conditioner_wtype_stat();
        std::map<ggml_type, uint32_t> diffusion_model_wtype_stat = model_loader.get_diffusion_model_wtype_stat();
        std::map<ggml_type, uint32_t> vae_wtype_stat             = model_loader.get_vae_wtype_stat();

        auto wtype_stat_to_str = [](const std::map<ggml_type, uint32_t>& m, int key_width = 8, int value_width = 5) -> std::string {
            std::ostringstream oss;
            bool first = true;
            for (const auto& [type, count] : m) {
                if (!first)
                    oss << "|";
                first = false;
                oss << std::right << std::setw(key_width) << ggml_type_name(type)
                    << ": "
                    << std::left << std::setw(value_width) << count;
            }
            return oss.str();
        };

        LOG_INFO("Weight type stat:                 %s", wtype_stat_to_str(wtype_stat).c_str());
        LOG_INFO("Conditioner weight type stat:     %s", wtype_stat_to_str(conditioner_wtype_stat).c_str());
        LOG_INFO("Diffusion model weight type stat: %s", wtype_stat_to_str(diffusion_model_wtype_stat).c_str());
        LOG_INFO("VAE weight type stat:             %s", wtype_stat_to_str(vae_wtype_stat).c_str());

        LOG_DEBUG("ggml tensor size = %d bytes", (int)sizeof(ggml_tensor));

        if (sd_ctx_params->lora_apply_mode == LORA_APPLY_AUTO) {
            bool have_quantized_weight = false;
            if (wtype != GGML_TYPE_COUNT && ggml_is_quantized(wtype)) {
                have_quantized_weight = true;
            } else {
                for (const auto& [type, _] : wtype_stat) {
                    if (ggml_is_quantized(type)) {
                        have_quantized_weight = true;
                        break;
                    }
                }
            }
            if (have_quantized_weight) {
                apply_lora_immediately = false;
            } else {
                apply_lora_immediately = true;
            }
        } else if (sd_ctx_params->lora_apply_mode == LORA_APPLY_IMMEDIATELY) {
            apply_lora_immediately = true;
        } else {
            apply_lora_immediately = false;
        }

        std::map<std::string, ggml_tensor*> mmap_able_tensors;
        bool enable_mmap_tensors = false;
        bool needs_writable_mmap = false;
        if (sd_ctx_params->enable_mmap) {
            if (apply_lora_immediately) {
                needs_writable_mmap = true;
                LOG_WARN("in mode 'immediately', LoRAs will cause extra memory usage with mmap");
            }
            enable_mmap_tensors = true;
        }

        // split definition to avoid msvc choking on the extra parameter handling
        auto module_can_mmap = [&](SDBackendModule module) {
            return enable_mmap_tensors &&
                   (backend_manager.runtime_backend_is_cpu(module) ||
                    backend_manager.params_backend_is_cpu(module) ||
                    backend_manager.runtime_backend_supports_host_buffer(module));
        };

        auto get_param_tensors_p = [&](auto&& model, bool do_mmap, const char* prefix) {
            std::map<std::string, ggml_tensor*> temp;
            model->get_param_tensors(temp, prefix);
            for (const auto& [key, tensor] : temp) {
                tensors[key] = tensor;
                if (do_mmap) {
                    mmap_able_tensors[key] = tensor;
                }
            }
        };

        auto get_param_tensors = [&](auto&& model, bool do_mmap) {
            std::map<std::string, ggml_tensor*> temp;
            model->get_param_tensors(temp);
            for (const auto& [key, tensor] : temp) {
                tensors[key] = tensor;
                if (do_mmap) {
                    mmap_able_tensors[key] = tensor;
                }
            }
        };

        if (sd_version_is_control(version)) {
            // Might need vae encode for control cond
            vae_decode_only = false;
        }
        bool tae_preview_only = sd_ctx_params->tae_preview_only;
        if (version == VERSION_SDXS_512_DS || version == VERSION_SDXS_09) {
            tae_preview_only = false;
            use_tae          = true;
        }

        if (sd_ctx_params->circular_x || sd_ctx_params->circular_y) {
            LOG_INFO("Using circular padding for convolutions");
        }

        const size_t max_graph_vram_bytes = sd::ggml_graph_cut::max_vram_gib_to_bytes(max_vram);

        {
            if (!ensure_backend_pair(SDBackendModule::TE) ||
                !ensure_backend_pair(SDBackendModule::DIFFUSION)) {
                return false;
            }

            if (sd_version_is_flux2(version)) {
                cond_stage_model = std::make_shared<LLMEmbedder>(backend_for(SDBackendModule::TE),
                                                                 params_backend_for(SDBackendModule::TE),
                                                                 tensor_storage_map,
                                                                 version);
                diffusion_model  = std::make_shared<Flux::FluxRunner>(backend_for(SDBackendModule::DIFFUSION),
                                                                     params_backend_for(SDBackendModule::DIFFUSION),
                                                                     tensor_storage_map,
                                                                     "model.diffusion_model",
                                                                     version,
                                                                     sd_ctx_params->chroma_use_dit_mask);
            }

            cond_stage_model->set_max_graph_vram_bytes(max_graph_vram_bytes);
            get_param_tensors(cond_stage_model, module_can_mmap(SDBackendModule::TE));

            diffusion_model->set_max_graph_vram_bytes(max_graph_vram_bytes);
            get_param_tensors(diffusion_model, module_can_mmap(SDBackendModule::DIFFUSION));

            if (sd_version_is_unet_edit(version)) {
                vae_decode_only = false;
            }

            if (!ensure_backend_pair(SDBackendModule::VAE)) {
                return false;
            }

            auto create_vae = [&]() -> std::shared_ptr<VAE> {
                auto model = std::make_shared<AutoEncoderKL>(backend_for(SDBackendModule::VAE),
                                                             params_backend_for(SDBackendModule::VAE),
                                                             tensor_storage_map,
                                                             "first_stage_model",
                                                             vae_decode_only,
                                                             false,
                                                             version);
                return model;
            };

            bool vae_mmap = module_can_mmap(SDBackendModule::VAE);

            LOG_INFO("using VAE for encoding / decoding");
            first_stage_model = create_vae();
            first_stage_model->set_max_graph_vram_bytes(max_graph_vram_bytes);
            get_param_tensors_p(first_stage_model, vae_mmap, "first_stage_model");

            if (sd_ctx_params->vae_conv_direct) {
                LOG_INFO("Using Conv2d direct in the vae model");
                first_stage_model->set_conv2d_direct_enabled(true);
                if (preview_vae) {
                    preview_vae->set_conv2d_direct_enabled(true);
                }
            }

            if (sd_ctx_params->flash_attn || sd_ctx_params->diffusion_flash_attn) {
                LOG_INFO("Using flash attention");
                cond_stage_model->set_flash_attention_enabled(true);
                diffusion_model->set_flash_attention_enabled(true);
            }

            if (first_stage_model) {
                first_stage_model->set_flash_attention_enabled(sd_ctx_params->flash_attn);
            }

            diffusion_model->set_circular_axes(sd_ctx_params->circular_x, sd_ctx_params->circular_y);
            circular_x = sd_ctx_params->circular_x;
            circular_y = sd_ctx_params->circular_y;
        }

        ggml_init_params params;
        params.mem_size   = static_cast<size_t>(10 * 1024) * 1024;  // 10M
        params.mem_buffer = nullptr;
        params.no_alloc   = false;
        // LOG_DEBUG("mem_size %u ", params.mem_size);
        ggml_context* ctx = ggml_init(params);  // for  alphas_cumprod and is_using_v_parameterization check
        GGML_ASSERT(ctx != nullptr);
        ggml_tensor* alphas_cumprod_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, TIMESTEPS);
        calculate_alphas_cumprod((float*)alphas_cumprod_tensor->data);

        // load weights
        LOG_DEBUG("loading weights");

        std::set<std::string> ignore_tensors;
        tensors["alphas_cumprod"] = alphas_cumprod_tensor;
        if (use_tae && !tae_preview_only) {
            ignore_tensors.insert("first_stage_model.");
        }
        if (vae_decode_only) {
            ignore_tensors.insert("first_stage_model.encoder");
            ignore_tensors.insert("first_stage_model.conv1");
            ignore_tensors.insert("first_stage_model.quant");
            ignore_tensors.insert("text_encoders.llm.visual.");
        }

        if (enable_mmap_tensors) {
            if (mmap_able_tensors.empty()) {
                LOG_DEBUG("no tensors could be memory-mapped");
            } else {
                mmap_tensor_store = model_loader.mmap_tensors(mmap_able_tensors, ignore_tensors, needs_writable_mmap);
            }
        }

        if (cond_stage_model) {
            cond_stage_model->alloc_params_buffer();
        }
        if (diffusion_model) {
            diffusion_model->alloc_params_buffer();
        }
        if (first_stage_model && !first_stage_model->alloc_params_buffer()) {
            LOG_ERROR("VAE params buffer allocation failed");
            ggml_free(ctx);
            return false;
        }
        if (preview_vae && !preview_vae->alloc_params_buffer()) {
            LOG_ERROR("preview VAE params buffer allocation failed");
            ggml_free(ctx);
            return false;
        }

        bool success = model_loader.load_tensors(tensors, ignore_tensors, n_threads, sd_ctx_params->enable_mmap);
        if (!success) {
            LOG_ERROR("load tensors from model loader failed");
            ggml_free(ctx);
            return false;
        }

        LOG_DEBUG("finished loaded file");

        {
            size_t clip_params_mem_size = cond_stage_model->get_params_buffer_size();
            size_t unet_params_mem_size = diffusion_model->get_params_buffer_size();
            size_t vae_params_mem_size  = first_stage_model->get_params_buffer_size();
            if (preview_vae) {
                vae_params_mem_size += preview_vae->get_params_buffer_size();
            }

            size_t total_params_ram_size  = 0;
            size_t total_params_vram_size = 0;
            auto add_params_memory        = [&](size_t size, SDBackendModule module) {
                if (size == 0) {
                    return true;
                }
                ggml_backend_t module_backend = params_backend_for(module);
                if (module_backend == nullptr) {
                    return false;
                }
                if (ggml_backend_is_cpu(module_backend)) {
                    total_params_ram_size += size;
                } else {
                    total_params_vram_size += size;
                }
                return true;
            };
            auto params_memory_location = [&](size_t size, SDBackendModule module) {
                if (size == 0) {
                    return "N/A";
                }
                ggml_backend_t module_backend = params_backend_for(module);
                if (module_backend == nullptr) {
                    return "N/A";
                }
                return ggml_backend_is_cpu(module_backend) ? "RAM" : "VRAM";
            };

            if (!add_params_memory(clip_params_mem_size, SDBackendModule::TE) ||
                !add_params_memory(unet_params_mem_size, SDBackendModule::DIFFUSION) ||
                !add_params_memory(vae_params_mem_size, SDBackendModule::VAE)) {
                ggml_free(ctx);
                return false;
            }

            size_t total_params_size = total_params_ram_size + total_params_vram_size;
            LOG_INFO(
                "total params memory size = %.2fMB (VRAM %.2fMB, RAM %.2fMB): "
                "text_encoders %.2fMB(%s), diffusion_model %.2fMB(%s), vae %.2fMB(%s)",
                total_params_size / 1024.0 / 1024.0,
                total_params_vram_size / 1024.0 / 1024.0,
                total_params_ram_size / 1024.0 / 1024.0,
                clip_params_mem_size / 1024.0 / 1024.0,
                params_memory_location(clip_params_mem_size, SDBackendModule::TE),
                unet_params_mem_size / 1024.0 / 1024.0,
                params_memory_location(unet_params_mem_size, SDBackendModule::DIFFUSION),
                vae_params_mem_size / 1024.0 / 1024.0,
                params_memory_location(vae_params_mem_size, SDBackendModule::VAE));
        }

        // init denoiser
        {
            prediction_t pred_type = sd_ctx_params->prediction;

            if (pred_type == PREDICTION_COUNT) {
                pred_type = FLUX2_FLOW_PRED;
            }

            if (pred_type != FLUX2_FLOW_PRED) {
                LOG_ERROR("Only FLUX2_FLOW_PRED is supported");
                ggml_free(ctx);
                return false;
            }

            LOG_INFO("running in Flux2 FLOW mode");
            denoiser = std::make_shared<Flux2FlowDenoiser>();
        }

        ggml_free(ctx);
        return true;
    }
    bool is_using_v_parameterization_for_sd2() {
        sd::Tensor<float> x_t   = sd::full<float>({8, 8, 4, 1}, 0.5f);
        sd::Tensor<float> c     = sd::full<float>({1024, 2, 1, 1}, 0.5f);
        sd::Tensor<float> steps = sd::full<float>({1}, 999.0f);
        sd::Tensor<float> concat;

        int64_t t0 = ggml_time_ms();
        sd::Tensor<float> out;
        DiffusionParams diffusion_params;
        diffusion_params.x         = &x_t;
        diffusion_params.timesteps = &steps;
        diffusion_params.context   = &c;
        diffusion_params.extra     = UNetDiffusionExtra{};
        if (!concat.empty()) {
            diffusion_params.c_concat = &concat;
        }
        auto out_opt = diffusion_model->compute(n_threads, diffusion_params);
        GGML_ASSERT(!out_opt.empty());
        out = std::move(out_opt);
        diffusion_model->free_compute_buffer();

        double result = static_cast<double>((out - x_t).mean());
        int64_t t1    = ggml_time_ms();
        LOG_DEBUG("check is_using_v_parameterization_for_sd2, taking %.2fs", (t1 - t0) * 1.0f / 1000);
        return result < -1;
    }

    std::shared_ptr<LoraModel> load_lora_model_from_file(const std::string& lora_id,
                                                         float multiplier,
                                                         SDBackendModule module,
                                                         LoraModel::filter_t lora_tensor_filter = nullptr) {
        std::string lora_path             = lora_id;
        static std::string high_noise_tag = "|high_noise|";
        bool is_high_noise                = false;
        if (starts_with(lora_path, high_noise_tag)) {
            lora_path     = lora_path.substr(high_noise_tag.size());
            is_high_noise = true;
            LOG_DEBUG("high noise lora: %s", lora_path.c_str());
        }
        if (!ensure_backend_pair(module)) {
            return nullptr;
        }
        auto lora = std::make_shared<LoraModel>(lora_id,
                                                backend_for(module),
                                                backend_for(module),
                                                lora_path,
                                                is_high_noise ? "model.high_noise_" : "",
                                                version);
        if (!lora->load_from_file(n_threads, lora_tensor_filter)) {
            LOG_WARN("load lora tensors from %s failed", lora_path.c_str());
            return nullptr;
        }

        lora->multiplier = multiplier;
        return lora;
    }

    void apply_loras_immediately(const std::unordered_map<std::string, float>& lora_state) {
        std::unordered_map<std::string, float> lora_state_diff;
        for (auto& kv : lora_state) {
            const std::string& lora_name = kv.first;
            float multiplier             = kv.second;
            lora_state_diff[lora_name] += multiplier;
        }
        for (auto& kv : curr_lora_state) {
            const std::string& lora_name = kv.first;
            float curr_multiplier        = kv.second;
            lora_state_diff[lora_name] -= curr_multiplier;
        }

        if (lora_state_diff.empty()) {
            return;
        }

        LOG_INFO("apply lora immediately");

        size_t rm = lora_state_diff.size() - lora_state.size();
        if (rm != 0) {
            LOG_INFO("attempting to apply %lu LoRAs (removing %lu applied LoRAs)", lora_state.size(), rm);
        } else {
            LOG_INFO("attempting to apply %lu LoRAs", lora_state.size());
        }

        for (auto& kv : lora_state_diff) {
            int64_t t0 = ggml_time_ms();

            auto lora = load_lora_model_from_file(kv.first, kv.second, SDBackendModule::DIFFUSION);
            if (!lora || lora->lora_tensors.empty()) {
                continue;
            }
            lora->apply(tensors, version, n_threads);
            lora->free_params_buffer();

            int64_t t1 = ggml_time_ms();

            LOG_INFO("lora '%s' applied, taking %.2fs", kv.first.c_str(), (t1 - t0) * 1.0f / 1000);
        }

        curr_lora_state = lora_state;
    }

    void apply_loras_at_runtime(const std::unordered_map<std::string, float>& lora_state) {
        cond_stage_lora_models.clear();
        diffusion_lora_models.clear();
        first_stage_lora_models.clear();
        if (cond_stage_model) {
            cond_stage_model->set_weight_adapter(nullptr);
        }
        if (diffusion_model) {
            diffusion_model->set_weight_adapter(nullptr);
        }
        if (first_stage_model) {
            first_stage_model->set_weight_adapter(nullptr);
        }
        if (lora_state.empty()) {
            return;
        }
        LOG_INFO("apply lora at runtime");
        if (cond_stage_model) {
            std::vector<std::shared_ptr<LoraModel>> lora_models;
            auto lora_state_diff = lora_state;
            for (auto& lora_model : cond_stage_lora_models) {
                auto iter = lora_state_diff.find(lora_model->lora_id);

                if (iter != lora_state_diff.end()) {
                    lora_model->multiplier = iter->second;
                    lora_models.push_back(lora_model);
                    lora_state_diff.erase(iter);
                }
            }
            cond_stage_lora_models  = lora_models;
            auto lora_tensor_filter = [&](const std::string& tensor_name) {
                if (is_cond_stage_model_name(tensor_name)) {
                    return true;
                }
                return false;
            };
            for (auto& kv : lora_state_diff) {
                const std::string& lora_id = kv.first;
                float multiplier           = kv.second;

                auto lora = load_lora_model_from_file(lora_id, multiplier, SDBackendModule::TE, lora_tensor_filter);
                if (lora && !lora->lora_tensors.empty()) {
                    lora->preprocess_lora_tensors(tensors);
                    cond_stage_lora_models.push_back(lora);
                }
            }
            // Only attach the adapter when there are LoRAs targeting the cond_stage model.
            // An empty MultiLoraAdapter still routes every linear/conv through
            // forward_with_lora() instead of the direct kernel path — slower for no benefit.
            if (!cond_stage_lora_models.empty()) {
                auto multi_lora_adapter = std::make_shared<MultiLoraAdapter>(cond_stage_lora_models);
                cond_stage_model->set_weight_adapter(multi_lora_adapter);
            }
        }
        if (diffusion_model) {
            std::vector<std::shared_ptr<LoraModel>> lora_models;
            auto lora_state_diff = lora_state;
            for (auto& lora_model : diffusion_lora_models) {
                auto iter = lora_state_diff.find(lora_model->lora_id);

                if (iter != lora_state_diff.end()) {
                    lora_model->multiplier = iter->second;
                    lora_models.push_back(lora_model);
                    lora_state_diff.erase(iter);
                }
            }
            diffusion_lora_models   = lora_models;
            auto lora_tensor_filter = [&](const std::string& tensor_name) {
                if (is_diffusion_model_name(tensor_name)) {
                    return true;
                }
                return false;
            };
            for (auto& kv : lora_state_diff) {
                const std::string& lora_name = kv.first;
                float multiplier             = kv.second;

                auto lora = load_lora_model_from_file(lora_name, multiplier, SDBackendModule::DIFFUSION, lora_tensor_filter);
                if (lora && !lora->lora_tensors.empty()) {
                    lora->preprocess_lora_tensors(tensors);
                    diffusion_lora_models.push_back(lora);
                }
            }
            if (!diffusion_lora_models.empty()) {
                auto multi_lora_adapter = std::make_shared<MultiLoraAdapter>(diffusion_lora_models);
                diffusion_model->set_weight_adapter(multi_lora_adapter);
            }
        }

        if (first_stage_model) {
            std::vector<std::shared_ptr<LoraModel>> lora_models;
            auto lora_state_diff = lora_state;
            for (auto& lora_model : first_stage_lora_models) {
                auto iter = lora_state_diff.find(lora_model->lora_id);

                if (iter != lora_state_diff.end()) {
                    lora_model->multiplier = iter->second;
                    lora_models.push_back(lora_model);
                    lora_state_diff.erase(iter);
                }
            }
            first_stage_lora_models = lora_models;
            auto lora_tensor_filter = [&](const std::string& tensor_name) {
                if (is_first_stage_model_name(tensor_name)) {
                    return true;
                }
                return false;
            };
            for (auto& kv : lora_state_diff) {
                const std::string& lora_name = kv.first;
                float multiplier             = kv.second;

                auto lora = load_lora_model_from_file(lora_name, multiplier, SDBackendModule::VAE, lora_tensor_filter);
                if (lora && !lora->lora_tensors.empty()) {
                    lora->preprocess_lora_tensors(tensors);
                    first_stage_lora_models.push_back(lora);
                }
            }
            if (!first_stage_lora_models.empty()) {
                auto multi_lora_adapter = std::make_shared<MultiLoraAdapter>(first_stage_lora_models);
                first_stage_model->set_weight_adapter(multi_lora_adapter);
            }
        }
    }

    void lora_stat() {
        if (!cond_stage_lora_models.empty()) {
            LOG_INFO("cond_stage_lora_models:");
            for (auto& lora_model : cond_stage_lora_models) {
                lora_model->stat();
            }
        }

        if (!diffusion_lora_models.empty()) {
            LOG_INFO("diffusion_lora_models:");
            for (auto& lora_model : diffusion_lora_models) {
                lora_model->stat();
            }
        }

        if (!first_stage_lora_models.empty()) {
            LOG_INFO("first_stage_lora_models:");
            for (auto& lora_model : first_stage_lora_models) {
                lora_model->stat();
            }
        }
    }

    void apply_loras(const sd_lora_t* loras, uint32_t lora_count) {
        std::unordered_map<std::string, float> lora_f2m;
        for (uint32_t i = 0; i < lora_count; i++) {
            std::string lora_id = SAFE_STR(loras[i].path);
            if (loras[i].is_high_noise) {
                lora_id = "|high_noise|" + lora_id;
            }
            lora_f2m[lora_id] = loras[i].multiplier;
            LOG_DEBUG("lora %s:%.2f", lora_id.c_str(), loras[i].multiplier);
        }
        int64_t t0 = ggml_time_ms();
        if (apply_lora_immediately) {
            apply_loras_immediately(lora_f2m);
        } else {
            apply_loras_at_runtime(lora_f2m);
        }
        int64_t t1 = ggml_time_ms();
        if (!lora_f2m.empty()) {
            LOG_INFO("apply_loras completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);
        }
    }

    SDCondition get_pmid_conditon(sd_pm_params_t pm_params,
                                  ConditionerParams& condition_params) {
        return SDCondition();
    }

    std::vector<float> process_timesteps(const std::vector<float>& timesteps,
                                         const sd::Tensor<float>& init_latent,
                                         const sd::Tensor<float>& denoise_mask) {
        return timesteps;
    }

    void preview_image(int step,
                       const sd::Tensor<float>& latents,
                       enum SDVersion version,
                       preview_t preview_mode,
                       std::function<void(int, int, sd_image_t*, bool, void*)> step_callback,
                       void* step_callback_data,
                       bool is_noisy) {
        int channels = get_latent_channel();
        auto _latents = channels != static_cast<int>(latents.shape()[2])
                            ? sd::ops::slice(latents, 2, 0, channels)
                            : latents;
        if (preview_mode == PREVIEW_PROJ) {
            const float(*latent_rgb_proj)[3] = flux2_latent_rgb_proj;
            float* latent_rgb_bias           = flux2_latent_rgb_bias;
            int patch_sz                     = 2;

            uint32_t img_width  = static_cast<uint32_t>(_latents.shape()[0]) * patch_sz;
            uint32_t img_height = static_cast<uint32_t>(_latents.shape()[1]) * patch_sz;

            uint32_t frames    = 1;
            uint8_t* data      = (uint8_t*)malloc(frames * img_width * img_height * 3 * sizeof(uint8_t));
            GGML_ASSERT(data != nullptr);
            preview_latent_video(data, _latents, latent_rgb_proj, latent_rgb_bias, patch_sz);
            sd_image_t images = {img_width, img_height, 3, data};
            step_callback(step, 1, &images, is_noisy, step_callback_data);
            free(data);
            return;
        }

        if (preview_mode == PREVIEW_VAE || preview_mode == PREVIEW_TAE) {
            sd::Tensor<float> decoded;
            if (preview_vae) {
                auto vae_latents = preview_vae->diffusion_to_vae_latents(_latents);
                decoded          = preview_vae->decode(n_threads, vae_latents, vae_tiling_params, false, circular_x, circular_y, true);
            } else {
                auto vae_latents = first_stage_model->diffusion_to_vae_latents(_latents);
                decoded          = first_stage_model->decode(n_threads, vae_latents, vae_tiling_params, false, circular_x, circular_y, true);
            }
            if (decoded.empty()) {
                LOG_ERROR("preview decode failed at step %d", step);
                return;
            }

            sd_image_t image = tensor_to_sd_image(decoded, 0);
            step_callback(step, 1, &image, is_noisy, step_callback_data);
            free(image.data);
            return;
        }

        if (preview_mode != PREVIEW_NONE) {
            LOG_WARN("Unsupported preview mode: %d", static_cast<int>(preview_mode));
        }
    }

    std::vector<float> prepare_sample_timesteps(float sigma,
                                                int shifted_timestep) {
        float t = denoiser->sigma_to_t(sigma);
        if (shifted_timestep > 0) {
            float shifted_t_float = t * (float(shifted_timestep) / float(TIMESTEPS));
            int64_t shifted_t     = static_cast<int64_t>(roundf(shifted_t_float));
            shifted_t             = std::max((int64_t)0, std::min((int64_t)(TIMESTEPS - 1), shifted_t));
            LOG_DEBUG("shifting timestep from %.2f to %" PRId64 " (sigma: %.4f)", t, shifted_t, sigma);
            return std::vector<float>{(float)shifted_t};
        }
        return std::vector<float>{t};
    }

    void adjust_sample_step_scalings(int shifted_timestep,
                                     const std::vector<float>& timesteps_vec,
                                     float c_in,
                                     float* c_skip,
                                     float* c_out) {
        GGML_ASSERT(c_skip != nullptr);
        GGML_ASSERT(c_out != nullptr);
        if (shifted_timestep <= 0) {
            return;
        }

        int64_t shifted_t_idx              = static_cast<int64_t>(roundf(timesteps_vec[0]));
        float shifted_sigma                = denoiser->t_to_sigma((float)shifted_t_idx);
        std::vector<float> shifted_scaling = denoiser->get_scalings(shifted_sigma);
        float shifted_c_skip               = shifted_scaling[0];
        float shifted_c_out                = shifted_scaling[1];
        float shifted_c_in                 = shifted_scaling[2];

        *c_skip = shifted_c_skip * c_in / shifted_c_in;
        *c_out  = shifted_c_out;
    }

    struct SamplePreviewContext {
        sd_preview_cb_t callback = nullptr;
        void* data               = nullptr;
        preview_t mode           = PREVIEW_NONE;
    };

    SamplePreviewContext prepare_sample_preview_context() {
        return SamplePreviewContext{sd_get_preview_callback(),
                                    sd_get_preview_callback_data(),
                                    sd_get_preview_mode()};
    }

    void report_sample_progress(int step, size_t total_steps, int64_t t0) {
        int64_t t1 = ggml_time_us();
        if (step > 0 || step == -(int)total_steps) {
            int showstep = std::abs(step);
            pretty_progress(showstep, (int)total_steps, (t1 - t0) / 1000000.f / showstep);
        }
    }

    sd::Tensor<float> sample(const std::shared_ptr<DiffusionModelRunner>& work_diffusion_model,
                             bool inverse_noise_scaling,
                             const sd::Tensor<float>& init_latent,
                             sd::Tensor<float> noise,
                             const SDCondition& cond,
                             const SDCondition& uncond,
                             const SDCondition& img_cond,
                             const SDCondition& id_cond,
                             const sd::Tensor<float>& control_image,
                             float control_strength,
                             const sd_guidance_params_t& guidance,
                             float eta,
                             int shifted_timestep,
                             sample_method_t method,
                             bool is_flow_denoiser,
                             const char* extra_sample_args,
                             const std::vector<float>& sigmas,
                             int start_merge_step,
                             const std::vector<sd::Tensor<float>>& ref_latents,
                             bool increase_ref_index,
                             const sd::Tensor<float>& denoise_mask,
                             const sd::Tensor<float>& vace_context,
                             float vace_strength,
                             int audio_length,
                             float frame_rate,
                             const sd_cache_params_t* cache_params,
                             const sd::Tensor<float>& video_positions = {}) {
        float cfg_scale     = guidance.txt_cfg;
        float img_cfg_scale = guidance.img_cfg;

        size_t steps = sigmas.size() - 1;
        sd::guidance::ClassifierFreeGuidance classifier_free_guidance(cfg_scale, img_cfg_scale);

        int64_t t0                   = ggml_time_us();
        sd::Tensor<float> x_t        = !noise.empty()
                                           ? denoiser->noise_scaling(sigmas[0], noise, init_latent)
                                           : init_latent;
        sd::Tensor<float> denoised   = x_t;
        SamplePreviewContext preview = prepare_sample_preview_context();

        auto denoise = [&](const sd::Tensor<float>& x, float sigma, int step) -> sd::guidance::GuiderOutput {
            if (step == 1 || step == -1) {
                pretty_progress(0, (int)steps, 0);
            }

            std::vector<float> scaling = denoiser->get_scalings(sigma);
            GGML_ASSERT(scaling.size() == 3);
            float c_skip = scaling[0];
            float c_out  = scaling[1];
            float c_in   = scaling[2];

            std::vector<float> timesteps_vec = prepare_sample_timesteps(sigma, shifted_timestep);
            adjust_sample_step_scalings(shifted_timestep, timesteps_vec, c_in, &c_skip, &c_out);

            sd::Tensor<float> timesteps_tensor({static_cast<int64_t>(timesteps_vec.size())}, timesteps_vec);
            sd::Tensor<float> guidance_tensor({1}, std::vector<float>{guidance.distilled_guidance});
            sd::Tensor<float> noised_input = x * c_in;

            if (sd_should_preview_noisy() && preview.callback != nullptr) {
                preview_image(step, noised_input, version, preview.mode, preview.callback, preview.data, true);
            }

            sd::Tensor<float> cond_out;
            sd::Tensor<float> uncond_out;
            sd::Tensor<float> img_cond_out;
            DiffusionParams diffusion_params;
            diffusion_params.x         = &noised_input;
            diffusion_params.timesteps = &timesteps_tensor;
            diffusion_params.extra     = FluxDiffusionExtra{&guidance_tensor, nullptr};

            auto run_condition = [&](const SDCondition& condition) -> sd::Tensor<float> {
                diffusion_params.context  = condition.c_crossattn.empty() ? nullptr : &condition.c_crossattn;
                diffusion_params.c_concat = condition.c_concat.empty() ? nullptr : &condition.c_concat;
                diffusion_params.y        = condition.c_vector.empty() ? nullptr : &condition.c_vector;

                auto output_opt = work_diffusion_model->compute(n_threads, diffusion_params);
                if (output_opt.empty()) {
                    LOG_ERROR("diffusion model compute failed");
                    return sd::Tensor<float>();
                }
                return output_opt;
            };

            cond_out = run_condition(cond);
            if (cond_out.empty()) {
                return {};
            }

            if (!uncond.empty()) {
                uncond_out = run_condition(uncond);
                if (uncond_out.empty()) {
                    return {};
                }
            }
            if (!img_cond.empty()) {
                img_cond_out = run_condition(img_cond);
                if (img_cond_out.empty()) {
                    return {};
                }
            }
            sd::guidance::GuidanceInput guidance_input;
            guidance_input.step          = step;
            guidance_input.schedule_size = sigmas.size();
            guidance_input.pred_cond     = &cond_out;
            guidance_input.pred_uncond   = uncond_out.empty() ? nullptr : &uncond_out;
            guidance_input.pred_img_cond = img_cond_out.empty() ? nullptr : &img_cond_out;

            sd::guidance::GuiderOutput guided = classifier_free_guidance.forward(guidance_input, {});
            if (guided.pred.empty()) {
                return {};
            }

            denoised = guided.pred * c_out + x * c_skip;
            sd::guidance::GuiderOutput output;
            output.pred = denoised;
            if (sd_should_preview_denoised() && preview.callback != nullptr) {
                preview_image(step, denoised, version, preview.mode, preview.callback, preview.data, false);
            }
            report_sample_progress(step, steps, t0);
            output.pred = denoised;
            return output;
        };

        auto x0_opt = sample_k_diffusion(method, denoise, x_t, sigmas, sampler_rng, eta, is_flow_denoiser, extra_sample_args);
        if (x0_opt.empty()) {
            LOG_ERROR("Diffusion model sampling failed");
            if (work_diffusion_model) {
                work_diffusion_model->free_compute_buffer();
            }
            return {};
        }

        auto x0 = std::move(x0_opt);
        if (inverse_noise_scaling) {
            x0 = denoiser->inverse_noise_scaling(sigmas[sigmas.size() - 1], x0);
        }

        if (work_diffusion_model) {
            work_diffusion_model->free_compute_buffer();
        }
        return x0;
    }

    int get_vae_scale_factor() {
        return first_stage_model->get_scale_factor();
    }

    int get_diffusion_model_down_factor() {
        return 1;
    }

    int get_latent_channel() {
        return 128;
    }

    int get_image_seq_len(int h, int w) {
        int vae_scale_factor = get_vae_scale_factor();
        return (h / vae_scale_factor) * (w / vae_scale_factor);
    }

    sd::Tensor<float> generate_init_latent(int width,
                                           int height) {
        int vae_scale_factor = get_vae_scale_factor();
        int W                = width / vae_scale_factor;
        int H                = height / vae_scale_factor;
        int C                = get_latent_channel();
        return sd::zeros<float>({W, H, C, 1});
    }

    sd::Tensor<float> encode_to_vae_latents(const sd::Tensor<float>& x) {
        auto latents = first_stage_model->encode(n_threads, x, vae_tiling_params, circular_x, circular_y);
        if (latents.empty()) {
            return {};
        }
        latents = first_stage_model->vae_output_to_latents(latents, rng);
        return latents;
    }

    sd::Tensor<float> encode_first_stage(const sd::Tensor<float>& x) {
        auto latents = encode_to_vae_latents(x);
        if (latents.empty()) {
            return {};
        }
        if (version != VERSION_SD1_PIX2PIX) {
            latents = first_stage_model->vae_to_diffusion_latents(latents);
        }
        return latents;
    }

    sd::Tensor<float> decode_first_stage(const sd::Tensor<float>& x, bool decode_video = false) {
        auto latents = first_stage_model->diffusion_to_vae_latents(x);
        first_stage_model->set_temporal_tiling_enabled(vae_tiling_params.temporal_tiling);
        return first_stage_model->decode(n_threads, latents, vae_tiling_params, decode_video, circular_x, circular_y);
    }

    void set_flow_shift(float flow_shift = INFINITY) {
        auto flow_denoiser = std::dynamic_pointer_cast<Flux2FlowDenoiser>(denoiser);
        if (flow_denoiser) {
            if (flow_shift == INFINITY) {
                flow_shift = default_flow_shift;
            }
            flow_denoiser->set_shift(flow_shift);
        }
    }

    bool is_flow_denoiser() {
        return true;
    }
};

/*================================================= SD API ==================================================*/

#define NONE_STR "NONE"

const char* sd_type_name(enum sd_type_t type) {
    if ((int)type < std::min<int>(SD_TYPE_COUNT, GGML_TYPE_COUNT)) {
        return ggml_type_name((ggml_type)type);
    }
    return NONE_STR;
}

enum sd_type_t str_to_sd_type(const char* str) {
    for (int i = 0; i < std::min<int>(SD_TYPE_COUNT, GGML_TYPE_COUNT); i++) {
        auto trait = ggml_get_type_traits((ggml_type)i);
        if (!strcmp(str, trait->type_name)) {
            return (enum sd_type_t)i;
        }
    }
    return SD_TYPE_COUNT;
}

const char* rng_type_to_str[] = {
    "std_default",
    "cuda",
    "cpu",
};

const char* sd_rng_type_name(enum rng_type_t rng_type) {
    if (rng_type < RNG_TYPE_COUNT) {
        return rng_type_to_str[rng_type];
    }
    return NONE_STR;
}

enum rng_type_t str_to_rng_type(const char* str) {
    for (int i = 0; i < RNG_TYPE_COUNT; i++) {
        if (!strcmp(str, rng_type_to_str[i])) {
            return (enum rng_type_t)i;
        }
    }
    return RNG_TYPE_COUNT;
}

const char* sample_method_to_str[] = {
    "euler",
    "euler_a",
    "heun",
    "dpm2",
    "dpm++2s_a",
    "dpm++2m",
    "dpm++2mv2",
    "ipndm",
    "ipndm_v",
    "lcm",
    "ddim_trailing",
    "tcd",
    "res_multistep",
    "res_2s",
    "er_sde",
    "euler_cfg_pp",
    "euler_a_cfg_pp",
    "euler_ge",
};

const char* sd_sample_method_name(enum sample_method_t sample_method) {
    if (sample_method < SAMPLE_METHOD_COUNT) {
        return sample_method_to_str[sample_method];
    }
    return NONE_STR;
}

enum sample_method_t str_to_sample_method(const char* str) {
    for (int i = 0; i < SAMPLE_METHOD_COUNT; i++) {
        if (!strcmp(str, sample_method_to_str[i])) {
            return (enum sample_method_t)i;
        }
    }
    return SAMPLE_METHOD_COUNT;
}

const char* scheduler_to_str[] = {
    "discrete",
    "karras",
    "exponential",
    "ays",
    "gits",
    "sgm_uniform",
    "simple",
    "smoothstep",
    "kl_optimal",
    "lcm",
    "bong_tangent",
    "ltx2",
};

const char* sd_scheduler_name(enum scheduler_t scheduler) {
    if (scheduler < SCHEDULER_COUNT) {
        return scheduler_to_str[scheduler];
    }
    return NONE_STR;
}

enum scheduler_t str_to_scheduler(const char* str) {
    for (int i = 0; i < SCHEDULER_COUNT; i++) {
        if (!strcmp(str, scheduler_to_str[i])) {
            return (enum scheduler_t)i;
        }
    }
    return SCHEDULER_COUNT;
}

const char* prediction_to_str[] = {
    "eps",
    "v",
    "edm_v",
    "sd3_flow",
    "flux_flow",
    "flux2_flow",
};

const char* sd_prediction_name(enum prediction_t prediction) {
    if (prediction < PREDICTION_COUNT) {
        return prediction_to_str[prediction];
    }
    return NONE_STR;
}

enum prediction_t str_to_prediction(const char* str) {
    for (int i = 0; i < PREDICTION_COUNT; i++) {
        if (!strcmp(str, prediction_to_str[i])) {
            return (enum prediction_t)i;
        }
    }
    return PREDICTION_COUNT;
}

const char* preview_to_str[] = {
    "none",
    "proj",
    "tae",
    "vae",
};

const char* sd_preview_name(enum preview_t preview) {
    if (preview < PREVIEW_COUNT) {
        return preview_to_str[preview];
    }
    return NONE_STR;
}

enum preview_t str_to_preview(const char* str) {
    for (int i = 0; i < PREVIEW_COUNT; i++) {
        if (!strcmp(str, preview_to_str[i])) {
            return (enum preview_t)i;
        }
    }
    return PREVIEW_COUNT;
}

const char* lora_apply_mode_to_str[] = {
    "auto",
    "immediately",
    "at_runtime",
};

const char* sd_lora_apply_mode_name(enum lora_apply_mode_t mode) {
    if (mode < LORA_APPLY_MODE_COUNT) {
        return lora_apply_mode_to_str[mode];
    }
    return NONE_STR;
}

enum lora_apply_mode_t str_to_lora_apply_mode(const char* str) {
    for (int i = 0; i < LORA_APPLY_MODE_COUNT; i++) {
        if (!strcmp(str, lora_apply_mode_to_str[i])) {
            return (enum lora_apply_mode_t)i;
        }
    }
    return LORA_APPLY_MODE_COUNT;
}

const char* hires_upscaler_to_str[] = {
    "None",
    "Latent",
    "Latent (nearest)",
    "Latent (nearest-exact)",
    "Latent (antialiased)",
    "Latent (bicubic)",
    "Latent (bicubic antialiased)",
    "Lanczos",
    "Nearest",
    "Model",
};

const char* sd_hires_upscaler_name(enum sd_hires_upscaler_t upscaler) {
    if (upscaler >= SD_HIRES_UPSCALER_NONE && upscaler < SD_HIRES_UPSCALER_COUNT) {
        return hires_upscaler_to_str[upscaler];
    }
    return NONE_STR;
}

enum sd_hires_upscaler_t str_to_sd_hires_upscaler(const char* str) {
    for (int i = 0; i < SD_HIRES_UPSCALER_COUNT; i++) {
        if (!strcmp(str, hires_upscaler_to_str[i])) {
            return (enum sd_hires_upscaler_t)i;
        }
    }
    return SD_HIRES_UPSCALER_COUNT;
}

void sd_cache_params_init(sd_cache_params_t* cache_params) {
    *cache_params                             = {};
    cache_params->mode                        = SD_CACHE_DISABLED;
    cache_params->reuse_threshold             = INFINITY;
    cache_params->start_percent               = 0.15f;
    cache_params->end_percent                 = 0.95f;
    cache_params->error_decay_rate            = 1.0f;
    cache_params->use_relative_threshold      = true;
    cache_params->reset_error_on_compute      = true;
    cache_params->Fn_compute_blocks           = 8;
    cache_params->Bn_compute_blocks           = 0;
    cache_params->residual_diff_threshold     = 0.08f;
    cache_params->max_warmup_steps            = 8;
    cache_params->max_cached_steps            = -1;
    cache_params->max_continuous_cached_steps = -1;
    cache_params->taylorseer_n_derivatives    = 1;
    cache_params->taylorseer_skip_interval    = 1;
    cache_params->scm_mask                    = nullptr;
    cache_params->scm_policy_dynamic          = true;
    cache_params->spectrum_w                  = 0.40f;
    cache_params->spectrum_m                  = 3;
    cache_params->spectrum_lam                = 1.0f;
    cache_params->spectrum_window_size        = 2;
    cache_params->spectrum_flex_window        = 0.50f;
    cache_params->spectrum_warmup_steps       = 4;
    cache_params->spectrum_stop_percent       = 0.9f;
}

void sd_ctx_params_init(sd_ctx_params_t* sd_ctx_params) {
    *sd_ctx_params                         = {};
    sd_ctx_params->vae_decode_only         = true;
    sd_ctx_params->free_params_immediately = true;
    sd_ctx_params->n_threads               = sd_get_num_physical_cores();
    sd_ctx_params->wtype                   = SD_TYPE_COUNT;
    sd_ctx_params->rng_type                = CUDA_RNG;
    sd_ctx_params->sampler_rng_type        = RNG_TYPE_COUNT;
    sd_ctx_params->prediction              = PREDICTION_COUNT;
    sd_ctx_params->lora_apply_mode         = LORA_APPLY_AUTO;
    sd_ctx_params->offload_params_to_cpu   = false;
    sd_ctx_params->max_vram                = 0.f;
    sd_ctx_params->enable_mmap             = false;
    sd_ctx_params->keep_clip_on_cpu        = false;
    sd_ctx_params->keep_control_net_on_cpu = false;
    sd_ctx_params->keep_vae_on_cpu         = false;
    sd_ctx_params->diffusion_flash_attn    = false;
    sd_ctx_params->circular_x              = false;
    sd_ctx_params->circular_y              = false;
    sd_ctx_params->chroma_use_dit_mask     = true;
    sd_ctx_params->chroma_use_t5_mask      = false;
    sd_ctx_params->chroma_t5_mask_pad      = 1;
    sd_ctx_params->backend                 = nullptr;
    sd_ctx_params->params_backend          = nullptr;
}

char* sd_ctx_params_to_str(const sd_ctx_params_t* sd_ctx_params) {
    char* buf = (char*)malloc(8192);
    if (!buf)
        return nullptr;
    buf[0] = '\0';

    snprintf(buf + strlen(buf), 8192 - strlen(buf),
             "model_path: %s\n"
             "clip_l_path: %s\n"
             "clip_g_path: %s\n"
             "t5xxl_path: %s\n"
             "llm_path: %s\n"
             "llm_vision_path: %s\n"
             "diffusion_model_path: %s\n"
             "vae_path: %s\n"
             "taesd_path: %s\n"
             "tensor_type_rules: %s\n"
             "vae_decode_only: %s\n"
             "free_params_immediately: %s\n"
             "n_threads: %d\n"
             "wtype: %s\n"
             "rng_type: %s\n"
             "sampler_rng_type: %s\n"
             "prediction: %s\n"
             "offload_params_to_cpu: %s\n"
             "max_vram: %.3f\n"
             "backend: %s\n"
             "params_backend: %s\n"
             "keep_clip_on_cpu: %s\n"
             "keep_vae_on_cpu: %s\n"
             "flash_attn: %s\n"
             "diffusion_flash_attn: %s\n"
             "circular_x: %s\n"
             "circular_y: %s\n"
             "chroma_use_dit_mask: %s\n"
             "chroma_use_t5_mask: %s\n"
             "chroma_t5_mask_pad: %d\n",
             SAFE_STR(sd_ctx_params->model_path),
             SAFE_STR(sd_ctx_params->clip_l_path),
             SAFE_STR(sd_ctx_params->clip_g_path),
             SAFE_STR(sd_ctx_params->t5xxl_path),
             SAFE_STR(sd_ctx_params->llm_path),
             SAFE_STR(sd_ctx_params->llm_vision_path),
             SAFE_STR(sd_ctx_params->diffusion_model_path),
             SAFE_STR(sd_ctx_params->vae_path),
             SAFE_STR(sd_ctx_params->taesd_path),
             SAFE_STR(sd_ctx_params->tensor_type_rules),
             BOOL_STR(sd_ctx_params->vae_decode_only),
             BOOL_STR(sd_ctx_params->free_params_immediately),
             sd_ctx_params->n_threads,
             sd_type_name(sd_ctx_params->wtype),
             sd_rng_type_name(sd_ctx_params->rng_type),
             sd_rng_type_name(sd_ctx_params->sampler_rng_type),
             sd_prediction_name(sd_ctx_params->prediction),
             BOOL_STR(sd_ctx_params->offload_params_to_cpu),
             sd_ctx_params->max_vram,
             SAFE_STR(sd_ctx_params->backend),
             SAFE_STR(sd_ctx_params->params_backend),
             BOOL_STR(sd_ctx_params->keep_clip_on_cpu),
             BOOL_STR(sd_ctx_params->keep_vae_on_cpu),
             BOOL_STR(sd_ctx_params->flash_attn),
             BOOL_STR(sd_ctx_params->diffusion_flash_attn),
             BOOL_STR(sd_ctx_params->circular_x),
             BOOL_STR(sd_ctx_params->circular_y),
             BOOL_STR(sd_ctx_params->chroma_use_dit_mask),
             BOOL_STR(sd_ctx_params->chroma_use_t5_mask),
             sd_ctx_params->chroma_t5_mask_pad);

    return buf;
}

void sd_sample_params_init(sd_sample_params_t* sample_params) {
    *sample_params                             = {};
    sample_params->guidance.txt_cfg            = 7.0f;
    sample_params->guidance.img_cfg            = INFINITY;
    sample_params->guidance.distilled_guidance = 3.5f;
    sample_params->guidance.slg.layer_count    = 0;
    sample_params->guidance.slg.layer_start    = 0.01f;
    sample_params->guidance.slg.layer_end      = 0.2f;
    sample_params->guidance.slg.scale          = 0.f;
    sample_params->scheduler                   = SCHEDULER_COUNT;
    sample_params->sample_method               = SAMPLE_METHOD_COUNT;
    sample_params->sample_steps                = 20;
    sample_params->eta                         = INFINITY;
    sample_params->custom_sigmas               = nullptr;
    sample_params->custom_sigmas_count         = 0;
    sample_params->flow_shift                  = INFINITY;
    sample_params->extra_sample_args           = nullptr;
}

char* sd_sample_params_to_str(const sd_sample_params_t* sample_params) {
    char* buf = (char*)malloc(4096);
    if (!buf)
        return nullptr;
    buf[0] = '\0';

    snprintf(buf + strlen(buf), 4096 - strlen(buf),
             "(txt_cfg: %.2f, "
             "img_cfg: %.2f, "
             "distilled_guidance: %.2f, "
             "slg.layer_count: %zu, "
             "slg.layer_start: %.2f, "
             "slg.layer_end: %.2f, "
             "slg.scale: %.2f, "
             "scheduler: %s, "
             "sample_method: %s, "
             "sample_steps: %d, "
             "eta: %.2f, "
             "shifted_timestep: %d, "
             "flow_shift: %.2f, "
             "extra_sample_args: %s)",
             sample_params->guidance.txt_cfg,
             std::isfinite(sample_params->guidance.img_cfg)
                 ? sample_params->guidance.img_cfg
                 : sample_params->guidance.txt_cfg,
             sample_params->guidance.distilled_guidance,
             sample_params->guidance.slg.layer_count,
             sample_params->guidance.slg.layer_start,
             sample_params->guidance.slg.layer_end,
             sample_params->guidance.slg.scale,
             sd_scheduler_name(sample_params->scheduler),
             sd_sample_method_name(sample_params->sample_method),
             sample_params->sample_steps,
             sample_params->eta,
             sample_params->shifted_timestep,
             sample_params->flow_shift,
             SAFE_STR(sample_params->extra_sample_args));

    return buf;
}

void sd_img_gen_params_init(sd_img_gen_params_t* sd_img_gen_params) {
    *sd_img_gen_params = {};
    sd_sample_params_init(&sd_img_gen_params->sample_params);
    sd_img_gen_params->clip_skip         = -1;
    sd_img_gen_params->ref_images_count  = 0;
    sd_img_gen_params->width             = 512;
    sd_img_gen_params->height            = 512;
    sd_img_gen_params->strength          = 0.75f;
    sd_img_gen_params->seed              = -1;
    sd_img_gen_params->batch_count       = 1;
    sd_img_gen_params->control_strength  = 0.9f;
    sd_img_gen_params->pm_params         = {nullptr, 0, nullptr, 20.f};
    sd_img_gen_params->vae_tiling_params = {false, false, 0, 0, 0.5f, 0.0f, 0.0f, nullptr};
    sd_cache_params_init(&sd_img_gen_params->cache);
    memset(&sd_img_gen_params->hires, 0, sizeof(sd_img_gen_params->hires));
    sd_img_gen_params->hires.enabled = false;
}

char* sd_img_gen_params_to_str(const sd_img_gen_params_t* sd_img_gen_params) {
    char* buf = (char*)malloc(4096);
    if (!buf)
        return nullptr;
    buf[0] = '\0';

    char* sample_params_str = sd_sample_params_to_str(&sd_img_gen_params->sample_params);

    snprintf(buf + strlen(buf), 4096 - strlen(buf),
             "prompt: %s\n"
             "negative_prompt: %s\n"
             "clip_skip: %d\n"
             "width: %d\n"
             "height: %d\n"
             "sample_params: %s\n"
             "strength: %.2f\n"
             "seed: %" PRId64
             "\n"
             "batch_count: %d\n"
             "ref_images_count: %d\n"
             "auto_resize_ref_image: %s\n"
             "increase_ref_index: %s\n"
             "control_strength: %.2f\n"
             "photo maker: {style_strength = %.2f, id_images_count = %d, id_embed_path = %s}\n"
             "VAE tiling: %s (temporal=%s, extra_tiling_args=%s)\n"
             "hires: {enabled=%s, upscaler=%s, model_path=%s, scale=%.2f, target=%dx%d, steps=%d, denoising_strength=%.2f}\n",
             SAFE_STR(sd_img_gen_params->prompt),
             SAFE_STR(sd_img_gen_params->negative_prompt),
             sd_img_gen_params->clip_skip,
             sd_img_gen_params->width,
             sd_img_gen_params->height,
             SAFE_STR(sample_params_str),
             sd_img_gen_params->strength,
             sd_img_gen_params->seed,
             sd_img_gen_params->batch_count,
             sd_img_gen_params->ref_images_count,
             BOOL_STR(sd_img_gen_params->auto_resize_ref_image),
             BOOL_STR(sd_img_gen_params->increase_ref_index),
             sd_img_gen_params->control_strength,
             sd_img_gen_params->pm_params.style_strength,
             sd_img_gen_params->pm_params.id_images_count,
             SAFE_STR(sd_img_gen_params->pm_params.id_embed_path),
             BOOL_STR(sd_img_gen_params->vae_tiling_params.enabled),
             BOOL_STR(sd_img_gen_params->vae_tiling_params.temporal_tiling),
             SAFE_STR(sd_img_gen_params->vae_tiling_params.extra_tiling_args),
             BOOL_STR(sd_img_gen_params->hires.enabled),
             sd_hires_upscaler_name(sd_img_gen_params->hires.upscaler),
             SAFE_STR(sd_img_gen_params->hires.model_path),
             sd_img_gen_params->hires.scale,
             sd_img_gen_params->hires.target_width,
             sd_img_gen_params->hires.target_height,
             sd_img_gen_params->hires.steps,
             sd_img_gen_params->hires.denoising_strength);
    const char* cache_mode_str = "disabled";
    if (sd_img_gen_params->cache.mode == SD_CACHE_EASYCACHE) {
        cache_mode_str = "easycache";
    } else if (sd_img_gen_params->cache.mode == SD_CACHE_UCACHE) {
        cache_mode_str = "ucache";
    }
    snprintf(buf + strlen(buf), 4096 - strlen(buf),
             "cache: %s (threshold=%.3f, start=%.2f, end=%.2f)\n",
             cache_mode_str,
             get_cache_reuse_threshold(sd_img_gen_params->cache),
             sd_img_gen_params->cache.start_percent,
             sd_img_gen_params->cache.end_percent);
    free(sample_params_str);
    return buf;
}

struct sd_ctx_t {
    StableDiffusionGGML* sd = nullptr;
};

sd_ctx_t* new_sd_ctx(const sd_ctx_params_t* sd_ctx_params) {
    sd_ctx_t* sd_ctx = (sd_ctx_t*)malloc(sizeof(sd_ctx_t));
    if (sd_ctx == nullptr) {
        return nullptr;
    }

    sd_ctx->sd = new StableDiffusionGGML();
    if (sd_ctx->sd == nullptr) {
        free(sd_ctx);
        return nullptr;
    }

    if (!sd_ctx->sd->init(sd_ctx_params)) {
        delete sd_ctx->sd;
        sd_ctx->sd = nullptr;
        free(sd_ctx);
        return nullptr;
    }
    return sd_ctx;
}

void free_sd_ctx(sd_ctx_t* sd_ctx) {
    if (sd_ctx->sd != nullptr) {
        delete sd_ctx->sd;
        sd_ctx->sd = nullptr;
    }
    free(sd_ctx);
}

void free_sd_audio(sd_audio_t* audio) {
    if (audio == nullptr) {
        return;
    }
    free(audio->data);
    audio->data = nullptr;
    free(audio);
}

SD_API bool sd_ctx_supports_image_generation(const sd_ctx_t* sd_ctx) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr) {
        return false;
    }
    return true;
}

SD_API bool sd_ctx_supports_video_generation(const sd_ctx_t* sd_ctx) {
    (void)sd_ctx;
    return false;
}

enum sample_method_t sd_get_default_sample_method(const sd_ctx_t* sd_ctx) {
    return EULER_SAMPLE_METHOD;
}

enum scheduler_t sd_get_default_scheduler(const sd_ctx_t* sd_ctx, enum sample_method_t sample_method) {
    if (sample_method == LCM_SAMPLE_METHOD || sample_method == TCD_SAMPLE_METHOD) {
        return LCM_SCHEDULER;
    } else if (sample_method == DDIM_TRAILING_SAMPLE_METHOD) {
        return SIMPLE_SCHEDULER;
    }
    return DISCRETE_SCHEDULER;
}

static int64_t resolve_seed(int64_t seed) {
    if (seed >= 0) {
        return seed;
    }
    srand((int)time(nullptr));
    return rand();
}

static enum sample_method_t resolve_sample_method(sd_ctx_t* sd_ctx, enum sample_method_t sample_method) {
    if (sample_method == SAMPLE_METHOD_COUNT) {
        return sd_get_default_sample_method(sd_ctx);
    }
    return sample_method;
}

static scheduler_t resolve_scheduler(sd_ctx_t* sd_ctx,
                                     scheduler_t scheduler,
                                     enum sample_method_t sample_method) {
    if (scheduler == SCHEDULER_COUNT) {
        return sd_get_default_scheduler(sd_ctx, sample_method);
    }
    return scheduler;
}

static float resolve_eta(sd_ctx_t* sd_ctx,
                         float eta,
                         enum sample_method_t sample_method) {
    if (eta == INFINITY) {
        switch (sample_method) {
            case DDIM_TRAILING_SAMPLE_METHOD:
            case TCD_SAMPLE_METHOD:
            case RES_MULTISTEP_SAMPLE_METHOD:
            case RES_2S_SAMPLE_METHOD:
                return 0.0f;
            case EULER_A_SAMPLE_METHOD:
            case DPMPP2S_A_SAMPLE_METHOD:
            case ER_SDE_SAMPLE_METHOD:
            case EULER_A_CFG_PP_SAMPLE_METHOD:
                return 1.0f;
            default:;
        }
        return 0.0f;
    }
    return eta;
}

struct GenerationRequest {
    std::string prompt;
    std::string negative_prompt;
    int width                                = -1;
    int height                               = -1;
    int clip_skip                            = -1;
    int vae_scale_factor                     = -1;
    int diffusion_model_down_factor          = -1;
    int64_t seed                             = -1;
    bool use_uncond   = false;
    bool use_img_cond = false;
    const sd_cache_params_t* cache_params    = nullptr;
    int batch_count                          = 1;
    int shifted_timestep                     = 0;
    float strength                           = 1.f;
    float control_strength                   = 0.f;
    float eta                                = 0.f;
    bool increase_ref_index                  = false;
    bool auto_resize_ref_image               = false;
    sd_guidance_params_t guidance = {};
    sd_pm_params_t pm_params      = {};
    sd_hires_params_t hires       = {};

    GenerationRequest(sd_ctx_t* sd_ctx, const sd_img_gen_params_t* sd_img_gen_params) {
        prompt                      = SAFE_STR(sd_img_gen_params->prompt);
        negative_prompt             = SAFE_STR(sd_img_gen_params->negative_prompt);
        width                       = sd_img_gen_params->width;
        height                      = sd_img_gen_params->height;
        vae_scale_factor            = sd_ctx->sd->get_vae_scale_factor();
        diffusion_model_down_factor = sd_ctx->sd->get_diffusion_model_down_factor();
        seed                        = sd_img_gen_params->seed;
        batch_count                 = sd_img_gen_params->batch_count;
        clip_skip                   = sd_img_gen_params->clip_skip;
        shifted_timestep            = sd_img_gen_params->sample_params.shifted_timestep;
        strength                    = sd_img_gen_params->strength;
        control_strength            = sd_img_gen_params->control_strength;
        eta                         = sd_img_gen_params->sample_params.eta;
        increase_ref_index          = sd_img_gen_params->increase_ref_index;
        auto_resize_ref_image       = sd_img_gen_params->auto_resize_ref_image;
        guidance                    = sd_img_gen_params->sample_params.guidance;
        pm_params                   = sd_img_gen_params->pm_params;
        hires                       = sd_img_gen_params->hires;
        cache_params                = &sd_img_gen_params->cache;
        resolve(sd_ctx);
    }

    GenerationRequest(sd_ctx_t* sd_ctx, const sd_vid_gen_params_t* sd_vid_gen_params) {
        GGML_UNUSED(sd_ctx);
        GGML_UNUSED(sd_vid_gen_params);
    }

    void align_generation_request_size() {
        align_image_size(&width, &height, "generation request");
    }

    void align_image_size(int* target_width, int* target_height, const char* label) {
        int spatial_multiple = vae_scale_factor * diffusion_model_down_factor;
        int width_offset     = align_up_offset(*target_width, spatial_multiple);
        int height_offset    = align_up_offset(*target_height, spatial_multiple);
        if (width_offset <= 0 && height_offset <= 0) {
            return;
        }

        int original_width  = *target_width;
        int original_height = *target_height;

        *target_width += width_offset;
        *target_height += height_offset;
        LOG_WARN("align %s up %dx%d to %dx%d (multiple=%d)",
                 label,
                 original_width,
                 original_height,
                 *target_width,
                 *target_height,
                 spatial_multiple);
    }

    static void resolve_guidance(sd_ctx_t* sd_ctx,
                                 sd_guidance_params_t* guidance,
                                 bool* use_uncond,
                                 bool* use_img_cond,
                                 const char* stage_name = nullptr) {
        GGML_ASSERT(guidance != nullptr);
        GGML_ASSERT(use_uncond != nullptr);
        GGML_ASSERT(use_img_cond != nullptr);
        // out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)
        // img_cfg == txt_cfg means that img_cfg is not used
        if (!std::isfinite(guidance->img_cfg)) {
            guidance->img_cfg = guidance->txt_cfg;
        }

        if (!sd_version_is_inpaint_or_unet_edit(sd_ctx->sd->version)) {
            guidance->img_cfg = guidance->txt_cfg;
        }

        if (guidance->txt_cfg != 1.f) {
            *use_uncond = true;
        }

        if (guidance->img_cfg != guidance->txt_cfg) {
            *use_img_cond = true;
            *use_uncond   = true;
        }

        if (guidance->txt_cfg < 1.f) {
            const char* prefix = stage_name == nullptr ? "" : stage_name;
            if (guidance->txt_cfg == 0.f) {
                LOG_WARN("%sunconditioned mode, images won't follow the prompt (use cfg-scale=1 for distilled models)",
                         prefix);
            } else {
                LOG_WARN("%scfg value out of expected range may produce unexpected results", prefix);
            }
        }
    }

    void resolve(sd_ctx_t* sd_ctx) {
        align_generation_request_size();
        seed = resolve_seed(seed);

        resolve_guidance(sd_ctx, &guidance, &use_uncond, &use_img_cond);

        if (shifted_timestep > 0) {
            LOG_WARN("timestep shifting is only supported for SDXL models!");
            shifted_timestep = 0;
        }
    }
};

struct SamplePlan {
    enum sample_method_t sample_method            = SAMPLE_METHOD_COUNT;
    enum sample_method_t high_noise_sample_method = SAMPLE_METHOD_COUNT;
    const char* extra_sample_args                 = nullptr;
    const char* high_noise_extra_sample_args      = nullptr;
    float eta                                     = 0.f;
    float high_noise_eta                          = 0.f;
    int sample_steps                              = 0;
    int high_noise_sample_steps                   = 0;
    int total_steps                               = 0;
    float moe_boundary                            = 0.f;
    int start_merge_step                          = -1;
    std::vector<float> sigmas;

    SamplePlan(sd_ctx_t* sd_ctx,
               const sd_img_gen_params_t* sd_img_gen_params,
               const GenerationRequest& request) {
        sample_method     = sd_img_gen_params->sample_params.sample_method;
        extra_sample_args = sd_img_gen_params->sample_params.extra_sample_args;
        eta               = sd_img_gen_params->sample_params.eta;
        sample_steps      = sd_img_gen_params->sample_params.sample_steps;
        resolve(sd_ctx, &request, &sd_img_gen_params->sample_params);
    }

    void resolve(sd_ctx_t* sd_ctx,
                 const GenerationRequest* request,
                 const sd_sample_params_t* sample_params) {
        sample_method = resolve_sample_method(sd_ctx, sample_method);

        total_steps = sample_steps + std::max(0, high_noise_sample_steps);

        if (sample_params->custom_sigmas_count > 0) {
            sigmas      = std::vector<float>(sample_params->custom_sigmas,
                                        sample_params->custom_sigmas + sample_params->custom_sigmas_count);
            total_steps = static_cast<int>(sigmas.size()) - 1;
            LOG_WARN("total_steps != custom_sigmas_count - 1, set total_steps to %d", total_steps);
            if (sample_steps >= total_steps) {
                sample_steps = total_steps;
                LOG_WARN("total_steps != custom_sigmas_count - 1, set sample_steps to %d", sample_steps);
            }
            if (high_noise_sample_steps > 0) {
                high_noise_sample_steps = total_steps - sample_steps;
                LOG_WARN("total_steps != custom_sigmas_count - 1, set high_noise_sample_steps to %d", high_noise_sample_steps);
            }
        } else {
            scheduler_t scheduler = resolve_scheduler(sd_ctx,
                                                      sample_params->scheduler,
                                                      sample_method);
            int sample_seq_len    = sd_ctx->sd->get_image_seq_len(request->height, request->width);
            sigmas = sd_ctx->sd->denoiser->get_sigmas(total_steps,
                                                      sample_seq_len,
                                                      scheduler,
                                                      sd_ctx->sd->version,
                                                      sample_params->extra_sample_args);
        }

        eta = resolve_eta(sd_ctx, eta, sample_method);

        if (high_noise_sample_steps < 0) {
            for (size_t i = 0; i < sigmas.size(); ++i) {
                if (sigmas[i] < moe_boundary) {
                    high_noise_sample_steps = static_cast<int>(i);
                    break;
                }
            }
            LOG_DEBUG("switching from high noise model at step %d", high_noise_sample_steps);
        }

        LOG_INFO("sampling using %s method", sampling_methods_str[sample_method]);
        if (high_noise_sample_steps > 0) {
            high_noise_sample_method = resolve_sample_method(sd_ctx,
                                                             high_noise_sample_method);
            high_noise_eta           = resolve_eta(sd_ctx, high_noise_eta, high_noise_sample_method);
            LOG_INFO("sampling(high noise) using %s method", sampling_methods_str[high_noise_sample_method]);
        }

        if (false) {
            // pmid not supported
        }
    }
};

struct ImageGenerationLatents {
    sd::Tensor<float> init_latent;
    sd::Tensor<float> concat_latent;
    sd::Tensor<float> uncond_concat_latent;
    sd::Tensor<float> audio_latent;
    sd::Tensor<float> video_positions;
    sd::Tensor<float> control_image;
    std::vector<sd::Tensor<float>> ref_images;
    std::vector<sd::Tensor<float>> ref_latents;
    sd::Tensor<float> denoise_mask;
    sd::Tensor<float> vace_context;
    int64_t ref_image_num                  = 0;
    int64_t video_conditioning_frame_count = 0;
    int64_t video_target_frame_count       = 0;
    int audio_length                       = 0;
};

struct ImageGenerationEmbeds {
    SDCondition cond;
    SDCondition uncond;
    SDCondition img_cond;
    SDCondition id_cond;
};

struct CircularAxesState {
    bool circular_x = false;
    bool circular_y = false;
};

static CircularAxesState configure_image_vae_axes(sd_ctx_t* sd_ctx,
                                                  const sd_img_gen_params_t* sd_img_gen_params,
                                                  const GenerationRequest& request) {
    CircularAxesState original_axes = {sd_ctx->sd->circular_x, sd_ctx->sd->circular_y};

    if (!sd_img_gen_params->vae_tiling_params.enabled) {
        if (sd_ctx->sd->first_stage_model) {
            sd_ctx->sd->first_stage_model->set_circular_axes(sd_ctx->sd->circular_x, sd_ctx->sd->circular_y);
        }
        if (sd_ctx->sd->preview_vae) {
            sd_ctx->sd->preview_vae->set_circular_axes(sd_ctx->sd->circular_x, sd_ctx->sd->circular_y);
        }
        return original_axes;
    }

    int tile_size_x, tile_size_y;
    float overlap;
    int latent_size_x = request.width / request.vae_scale_factor;
    int latent_size_y = request.height / request.vae_scale_factor;
    sd_ctx->sd->first_stage_model->get_tile_sizes(tile_size_x,
                                                  tile_size_y,
                                                  overlap,
                                                  sd_img_gen_params->vae_tiling_params,
                                                  latent_size_x,
                                                  latent_size_y);

    sd_ctx->sd->circular_x = sd_ctx->sd->circular_x && (tile_size_x >= latent_size_x);
    sd_ctx->sd->circular_y = sd_ctx->sd->circular_y && (tile_size_y >= latent_size_y);

    if (sd_ctx->sd->first_stage_model) {
        sd_ctx->sd->first_stage_model->set_circular_axes(sd_ctx->sd->circular_x, sd_ctx->sd->circular_y);
    }
    if (sd_ctx->sd->preview_vae) {
        sd_ctx->sd->preview_vae->set_circular_axes(sd_ctx->sd->circular_x, sd_ctx->sd->circular_y);
    }

    sd_ctx->sd->circular_x = original_axes.circular_x && (tile_size_x < latent_size_x);
    sd_ctx->sd->circular_y = original_axes.circular_y && (tile_size_y < latent_size_y);

    return original_axes;
}

static void restore_image_vae_axes(sd_ctx_t* sd_ctx, const CircularAxesState& original_axes) {
    sd_ctx->sd->circular_x = original_axes.circular_x;
    sd_ctx->sd->circular_y = original_axes.circular_y;
}

class ImageVaeAxesGuard {
private:
    sd_ctx_t* sd_ctx = nullptr;
    CircularAxesState original_axes;

public:
    ImageVaeAxesGuard(sd_ctx_t* sd_ctx,
                      const sd_img_gen_params_t* sd_img_gen_params,
                      const GenerationRequest& request)
        : sd_ctx(sd_ctx),
          original_axes(configure_image_vae_axes(sd_ctx, sd_img_gen_params, request)) {}

    ~ImageVaeAxesGuard() {
        restore_image_vae_axes(sd_ctx, original_axes);
    }

    ImageVaeAxesGuard(const ImageVaeAxesGuard&)            = delete;
    ImageVaeAxesGuard& operator=(const ImageVaeAxesGuard&) = delete;
};

static std::optional<ImageGenerationLatents> prepare_image_generation_latents(sd_ctx_t* sd_ctx,
                                                                              const sd_img_gen_params_t* sd_img_gen_params,
                                                                              GenerationRequest* request,
                                                                              SamplePlan* plan) {
    int64_t prepare_start_ms = ggml_time_ms();

    sd::Tensor<float> init_image_tensor;
    sd::Tensor<float> control_image_tensor;
    sd::Tensor<float> mask_image_tensor;

    if (sd_img_gen_params->init_image.data != nullptr) {
        LOG_INFO("IMG2IMG");

        if (request->strength < 1.f) {
            size_t t_enc = static_cast<size_t>(plan->sample_steps * request->strength);
            if (t_enc == static_cast<size_t>(plan->sample_steps)) {
                t_enc--;
            }
            LOG_INFO("target t_enc is %zu steps", t_enc);
            std::vector<float> sigma_sched;
            sigma_sched.assign(plan->sigmas.begin() + plan->sample_steps - t_enc - 1, plan->sigmas.end());
            plan->sigmas       = std::move(sigma_sched);
            plan->sample_steps = static_cast<int>(plan->sigmas.size() - 1);
        }

        init_image_tensor = sd_image_to_tensor(sd_img_gen_params->init_image, request->width, request->height);
    }

    if (sd_img_gen_params->mask_image.data != nullptr) {
        mask_image_tensor = sd_image_to_tensor(sd_img_gen_params->mask_image, request->width, request->height);
        mask_image_tensor = sd::ops::round(mask_image_tensor);
    }

    if (sd_img_gen_params->control_image.data != nullptr) {
        control_image_tensor = sd_image_to_tensor(sd_img_gen_params->control_image, request->width, request->height);
    }

    if (init_image_tensor.empty() || mask_image_tensor.empty()) {
        if (sd_version_is_inpaint(sd_ctx->sd->version)) {
            LOG_WARN("inpainting model requires both an init image and a mask image.");
        }
    }

    if (mask_image_tensor.empty()) {
        mask_image_tensor = sd::full<float>({request->width, request->height, 1, 1}, 1.f);
    }

    sd::Tensor<float> latent_mask = sd::ops::interpolate(mask_image_tensor,
                                                         {request->width / request->vae_scale_factor,
                                                          request->height / request->vae_scale_factor,
                                                          1,
                                                          1},
                                                         sd::ops::InterpolateMode::NearestMax);

    sd::Tensor<float> init_latent;
    sd::Tensor<float> control_latent;
    if (init_image_tensor.empty()) {
        init_latent = sd_ctx->sd->generate_init_latent(request->width, request->height);
    } else {
        init_latent = sd_ctx->sd->encode_first_stage(init_image_tensor);
        if (init_latent.empty()) {
            LOG_ERROR("failed to encode init image");
            return std::nullopt;
        }
    }

    if (!control_image_tensor.empty() && !sd_ctx->sd->vae_decode_only) {
        control_latent = sd_ctx->sd->encode_first_stage(control_image_tensor);
        if (control_latent.empty()) {
            LOG_ERROR("failed to encode control image");
            return std::nullopt;
        }
    }

    std::vector<sd::Tensor<float>> ref_images;
    for (int i = 0; i < sd_img_gen_params->ref_images_count; i++) {
        ref_images.push_back(sd_image_to_tensor(sd_img_gen_params->ref_images[i]));
    }

    if (ref_images.empty() && sd_version_is_unet_edit(sd_ctx->sd->version)) {
        LOG_WARN("This model needs at least one reference image; using an empty reference");
        ref_images.push_back(sd::zeros<float>({request->width, request->height, 3, 1}));
        request->guidance.img_cfg = request->guidance.txt_cfg;
    }

    if (!ref_images.empty()) {
        LOG_INFO("EDIT mode");
    }

    std::vector<sd::Tensor<float>> ref_latents;
    for (size_t i = 0; i < ref_images.size(); i++) {
        if (sd_ctx->sd->version == VERSION_HIDREAM_O1) {
            continue;
        }
        sd::Tensor<float> ref_latent;
        if (request->auto_resize_ref_image) {
            LOG_DEBUG("auto resize ref images");
            int vae_image_size = std::min(1024 * 1024, request->width * request->height);
            double vae_width   = sqrt(vae_image_size * ref_images[i].shape()[0] / ref_images[i].shape()[1]);
            double vae_height  = vae_width * ref_images[i].shape()[1] / ref_images[i].shape()[0];

            int factor = 32;
            vae_height = round(vae_height / factor) * factor;
            vae_width  = round(vae_width / factor) * factor;

            auto resized_ref_img = sd::ops::interpolate(ref_images[i],
                                                        {static_cast<int>(vae_width), static_cast<int>(vae_height), 3, 1});

            LOG_DEBUG("resize vae ref image %d from %" PRId64 "x%" PRId64 " to %" PRId64 "x%" PRId64,
                      static_cast<int>(i),
                      ref_images[i].shape()[1],
                      ref_images[i].shape()[0],
                      resized_ref_img.shape()[1],
                      resized_ref_img.shape()[0]);

            ref_latent = sd_ctx->sd->encode_first_stage(resized_ref_img);
        } else {
            ref_latent = sd_ctx->sd->encode_first_stage(ref_images[i]);
        }
        if (ref_latent.empty()) {
            LOG_ERROR("failed to encode reference image %d", static_cast<int>(i));
            return std::nullopt;
        }

        ref_latents.push_back(std::move(ref_latent));
    }

    sd::Tensor<float> concat_latent;
    sd::Tensor<float> uncond_concat_latent;
    if (sd_version_is_inpaint(sd_ctx->sd->version)) {
        sd::Tensor<float> masked_init_latent;

        if (sd_ctx->sd->version != VERSION_FLEX_2) {
            if (!init_image_tensor.empty()) {
                auto masked_image  = ((1.0f - mask_image_tensor) * (init_image_tensor - 0.5f)) + 0.5f;
                masked_init_latent = sd_ctx->sd->encode_first_stage(masked_image);
                if (masked_init_latent.empty()) {
                    LOG_ERROR("failed to encode masked init image");
                    return std::nullopt;
                }
            } else {
                masked_init_latent = sd::Tensor<float>::zeros_like(init_latent);
            }
        } else {
            masked_init_latent = ((1.0f - latent_mask) * init_latent);
        }

        auto uncond_masked_init_latent = sd::Tensor<float>::zeros_like(masked_init_latent);

        if (sd_ctx->sd->version == VERSION_FLUX_FILL) {
            auto mask = mask_image_tensor.reshape({request->vae_scale_factor,
                                                   request->width / request->vae_scale_factor,
                                                   request->vae_scale_factor,
                                                   request->height / request->vae_scale_factor});
            mask      = mask.permute({1, 3, 0, 2}).reshape({request->width / request->vae_scale_factor, request->height / request->vae_scale_factor, request->vae_scale_factor * request->vae_scale_factor, 1});

            concat_latent        = sd::ops::concat(masked_init_latent, mask, 2);
            uncond_concat_latent = sd::ops::concat(uncond_masked_init_latent, mask, 2);
        } else if (sd_ctx->sd->version == VERSION_FLEX_2) {
            concat_latent = sd::ops::concat(masked_init_latent, latent_mask, 2);
            if (!control_latent.empty()) {
                concat_latent = sd::ops::concat(concat_latent, control_latent, 2);
            } else {
                concat_latent = sd::ops::concat(concat_latent, sd::Tensor<float>::zeros_like(masked_init_latent), 2);
            }

            uncond_concat_latent = sd::ops::concat(uncond_masked_init_latent, latent_mask, 2);
            uncond_concat_latent = sd::ops::concat(uncond_concat_latent, sd::Tensor<float>::zeros_like(masked_init_latent), 2);
        } else {  // SD1.x SD2.x SDXL inpaint
            concat_latent        = sd::ops::concat(latent_mask, masked_init_latent, 2);
            uncond_concat_latent = sd::ops::concat(latent_mask, uncond_masked_init_latent, 2);
        }
    }
    if (sd_version_is_unet_edit(sd_ctx->sd->version)) {
        concat_latent        = sd::ops::interpolate<float>(ref_latents[0], init_latent.shape());
        uncond_concat_latent = sd::Tensor<float>::zeros_like(concat_latent);
    }
    if (sd_ctx->sd->version == VERSION_FLUX_CONTROLS) {
        if (!control_latent.empty()) {
            concat_latent = control_latent;
        } else {
            concat_latent = sd::Tensor<float>::zeros_like(init_latent);
        }
        uncond_concat_latent = sd::Tensor<float>::zeros_like(concat_latent);
    }

    if (sd_img_gen_params->init_image.data != nullptr || sd_img_gen_params->ref_images_count > 0) {
        int64_t t1 = ggml_time_ms();
        LOG_INFO("encode_first_stage completed, taking %.2fs", (t1 - prepare_start_ms) * 1.0f / 1000);
    }

    ImageGenerationLatents latents;
    latents.init_latent          = std::move(init_latent);
    latents.concat_latent        = std::move(concat_latent);
    latents.uncond_concat_latent = std::move(uncond_concat_latent);
    latents.control_image        = std::move(control_image_tensor);
    latents.ref_images           = std::move(ref_images);
    latents.ref_latents          = std::move(ref_latents);

    if (sd_version_is_inpaint(sd_ctx->sd->version)) {
        latent_mask = sd::ops::max_pool_2d(latent_mask,
                                           {3, 3},
                                           {1, 1},
                                           {1, 1});
    }
    latents.denoise_mask = std::move(latent_mask);

    return latents;
}

static std::optional<ImageGenerationEmbeds> prepare_image_generation_embeds(sd_ctx_t* sd_ctx,
                                                                            const sd_img_gen_params_t* sd_img_gen_params,
                                                                            GenerationRequest* request,
                                                                            SamplePlan* plan,
                                                                            ImageGenerationLatents* latents) {
    ConditionerParams condition_params;
    condition_params.text       = request->prompt;
    condition_params.clip_skip  = request->clip_skip;
    condition_params.width      = request->width;
    condition_params.height     = request->height;
    condition_params.ref_images = &latents->ref_images;

    auto id_cond                     = sd_ctx->sd->get_pmid_conditon(request->pm_params, condition_params);
    int64_t prepare_start_ms         = ggml_time_ms();
    condition_params.zero_out_masked = false;
    auto cond                        = sd_ctx->sd->cond_stage_model->get_learned_condition(sd_ctx->sd->n_threads,
                                                                                           condition_params);
    if (cond.c_concat.empty()) {
        cond.c_concat = latents->concat_latent;  // TODO: optimize
    }

    SDCondition uncond;
    if (request->use_uncond) {
        bool zero_out_masked = false;
        if (false) {
            zero_out_masked = true;
        }
        condition_params.text            = request->negative_prompt;
        condition_params.zero_out_masked = zero_out_masked;
        uncond                           = sd_ctx->sd->cond_stage_model->get_learned_condition(sd_ctx->sd->n_threads,
                                                                                               condition_params);
        if (uncond.c_concat.empty()) {
            uncond.c_concat = latents->uncond_concat_latent;  // TODO: optimize
        }
    }

    int64_t t1 = ggml_time_ms();
    LOG_INFO("get_learned_condition completed, taking %.2fs", (t1 - prepare_start_ms) * 1.0f / 1000);

    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->cond_stage_model->free_params_buffer();
    }

    ImageGenerationEmbeds embeds;
    if (request->use_img_cond) {
        embeds.img_cond = SDCondition(uncond.c_crossattn, uncond.c_vector, cond.c_concat);
    }
    embeds.cond    = std::move(cond);
    embeds.uncond  = std::move(uncond);
    embeds.id_cond = std::move(id_cond);

    return embeds;
}

static sd_image_t* decode_image_outputs(sd_ctx_t* sd_ctx,
                                        const GenerationRequest& request,
                                        const std::vector<sd::Tensor<float>>& final_latents) {
    if (final_latents.size() != static_cast<size_t>(request.batch_count)) {
        LOG_ERROR("expected %d latents, got %zu", request.batch_count, final_latents.size());
        return nullptr;
    }
    LOG_INFO("decoding %zu latents", final_latents.size());
    std::vector<sd::Tensor<float>> decoded_images;
    int64_t t0 = ggml_time_ms();

    for (size_t i = 0; i < final_latents.size(); i++) {
        int64_t t1              = ggml_time_ms();
        sd::Tensor<float> image = sd_ctx->sd->decode_first_stage(final_latents[i]);
        if (image.empty()) {
            LOG_ERROR("decode_first_stage failed for latent %" PRId64, i + 1);
            if (sd_ctx->sd->free_params_immediately) {
                sd_ctx->sd->first_stage_model->free_params_buffer();
            }
            return nullptr;
        }
        decoded_images.push_back(std::move(image));
        int64_t t2 = ggml_time_ms();
        LOG_INFO("latent %zu decoded, taking %.2fs", i + 1, (t2 - t1) * 1.0f / 1000);
    }

    int64_t t4 = ggml_time_ms();
    LOG_INFO("decode_first_stage completed, taking %.2fs", (t4 - t0) * 1.0f / 1000);
    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->first_stage_model->free_params_buffer();
    }

    sd_image_t* result_images = (sd_image_t*)calloc(request.batch_count, sizeof(sd_image_t));
    if (result_images == nullptr) {
        return nullptr;
    }
    memset(result_images, 0, request.batch_count * sizeof(sd_image_t));

    for (size_t i = 0; i < decoded_images.size(); i++) {
        result_images[i] = tensor_to_sd_image(decoded_images[i]);
    }

    return result_images;
}

SD_API sd_image_t* generate_image(sd_ctx_t* sd_ctx, const sd_img_gen_params_t* sd_img_gen_params) {
    if (sd_ctx == nullptr || sd_img_gen_params == nullptr) {
        return nullptr;
    }

    int64_t t0                    = ggml_time_ms();
    sd_ctx->sd->vae_tiling_params = sd_img_gen_params->vae_tiling_params;
    GenerationRequest request(sd_ctx, sd_img_gen_params);
    LOG_INFO("generate_image %dx%d", request.width, request.height);

    sd_ctx->sd->rng->manual_seed(request.seed);
    sd_ctx->sd->sampler_rng->manual_seed(request.seed);
    sd_ctx->sd->set_flow_shift(sd_img_gen_params->sample_params.flow_shift);
    sd_ctx->sd->apply_loras(sd_img_gen_params->loras, sd_img_gen_params->lora_count);

    ImageVaeAxesGuard axes_guard(sd_ctx, sd_img_gen_params, request);

    SamplePlan plan(sd_ctx, sd_img_gen_params, request);
    auto latents_opt = prepare_image_generation_latents(sd_ctx,
                                                        sd_img_gen_params,
                                                        &request,
                                                        &plan);
    if (!latents_opt.has_value()) {
        return nullptr;
    }
    ImageGenerationLatents latents = std::move(*latents_opt);

    auto embeds_opt = prepare_image_generation_embeds(sd_ctx,
                                                      sd_img_gen_params,
                                                      &request,
                                                      &plan,
                                                      &latents);
    if (!embeds_opt.has_value()) {
        return nullptr;
    }
    ImageGenerationEmbeds embeds = std::move(*embeds_opt);

    std::vector<sd::Tensor<float>> final_latents;
    int64_t denoise_start = ggml_time_ms();
    for (int b = 0; b < request.batch_count; b++) {
        int64_t sampling_start = ggml_time_ms();
        int64_t cur_seed       = request.seed + b;
        LOG_INFO("generating image: %i/%i - seed %" PRId64, b + 1, request.batch_count, cur_seed);

        sd_ctx->sd->rng->manual_seed(cur_seed);
        sd_ctx->sd->sampler_rng->manual_seed(cur_seed);
        sd::Tensor<float> noise = sd::randn_like<float>(latents.init_latent, sd_ctx->sd->rng);

        sd::Tensor<float> x_0 = sd_ctx->sd->sample(sd_ctx->sd->diffusion_model,
                                                   true,
                                                   latents.init_latent,
                                                   std::move(noise),
                                                   embeds.cond,
                                                   embeds.uncond,
                                                   embeds.img_cond,
                                                   embeds.id_cond,
                                                   latents.control_image,
                                                   request.control_strength,
                                                   request.guidance,
                                                   plan.eta,
                                                   request.shifted_timestep,
                                                   plan.sample_method,
                                                   sd_ctx->sd->is_flow_denoiser(),
                                                   plan.extra_sample_args,
                                                   plan.sigmas,
                                                   plan.start_merge_step,
                                                   latents.ref_latents,
                                                   request.increase_ref_index,
                                                   latents.denoise_mask,
                                                   sd::Tensor<float>(),
                                                   1.f,
                                                   0,
                                                    0.f,
                                                   request.cache_params);
        int64_t sampling_end  = ggml_time_ms();
        if (!x_0.empty()) {
            LOG_INFO("sampling completed, taking %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);
            final_latents.push_back(std::move(x_0));
            continue;
        }

        LOG_ERROR("sampling for image %d/%d failed after %.2fs",
                  b + 1,
                  request.batch_count,
                  (sampling_end - sampling_start) * 1.0f / 1000);
        if (sd_ctx->sd->free_params_immediately) {
            sd_ctx->sd->diffusion_model->free_params_buffer();
        }
        return nullptr;
    }
    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->diffusion_model->free_params_buffer();
    }
    int64_t denoise_end = ggml_time_ms();
    LOG_INFO("generating %zu latent images completed, taking %.2fs",
             final_latents.size(),
             (denoise_end - denoise_start) * 1.0f / 1000);

    auto result = decode_image_outputs(sd_ctx, request, final_latents);
    if (result == nullptr) {
        return nullptr;
    }

    sd_ctx->sd->lora_stat();

    int64_t t1 = ggml_time_ms();
    LOG_INFO("generate_image completed in %.2fs", (t1 - t0) * 1.0f / 1000);
    return result;
}

SD_API bool generate_video(sd_ctx_t* sd_ctx,
                           const sd_vid_gen_params_t* sd_vid_gen_params,
                           sd_image_t** frames_out,
                           int* num_frames_out,
                           sd_audio_t** audio_out) {
    (void)frames_out;
    (void)num_frames_out;
    (void)audio_out;
    LOG_ERROR("generate_video is not supported in this build (Flux2 only)");
    return false;
}

SD_API void sd_vid_gen_params_init(sd_vid_gen_params_t* sd_vid_gen_params) {
    (void)sd_vid_gen_params;
}

SD_API void sd_hires_params_init(sd_hires_params_t* hires_params) {
    (void)hires_params;
}

