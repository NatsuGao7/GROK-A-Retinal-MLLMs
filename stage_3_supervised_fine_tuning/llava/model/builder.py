#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    if 'llava' in model_name.lower() or 'grok' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            if "llama" in model_name.lower() or "vicuna" in model_name.lower():
                from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig
                model_class = LlavaLlamaForCausalLM
                config_class = LlavaConfig
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                print(f'Loading LLaVA with Llama base model...')
                # model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=False, config=lora_cfg_pretrained, **kwargs)


            elif "qwen" in model_name.lower():
                from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM, LlavaQwenConfig
                model_class = LlavaQwenForCausalLM
                config_class = LlavaQwenConfig
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                print(f'Loading LLaVA with Qwen base model...')
                # model = LlavaQwenForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=False, config=lora_cfg_pretrained, **kwargs)

            # 第 1 步：分别加载 LoRA 的目标配置和基础模型的真实配置
            lora_cfg_pretrained = config_class.from_pretrained(model_path)
            base_model_config = config_class.from_pretrained(model_base)
            
            # 第 2 步：【“欺骗”加载器】临时将 LoRA 配置的 vocab_size 修改为与基础模型一致
            original_lora_vocab_size = lora_cfg_pretrained.vocab_size # 先保存 LoRA 原始的 vocab_size
            lora_cfg_pretrained.vocab_size = base_model_config.vocab_size
            
            print('Loading LLaVA from base model with temporarily matched vocab size...')
            # 第 3 步：【安全加载】使用修改后的配置加载模型
            model = model_class.from_pretrained(
                model_base,
                low_cpu_mem_usage=False,  # 禁用 meta tensor 优化以兼容自定义模块
                config=lora_cfg_pretrained, 
                **kwargs
            )
            
            # 第 4 步：【恢复并调整】将模型的词嵌入层调整回 LoRA 期望的原始大小
            print(f"Resizing token embeddings back to LoRA's original size: {original_lora_vocab_size}...")
            model.resize_token_embeddings(original_lora_vocab_size)

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            elif "qwen" in model_name.lower() or "quyen" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                from ..model.language_model.llava_qwen import LlavaQwenConfig
                overwrite_config=None
                attn_implementation="flash_attention_2"
                if overwrite_config is not None:
                    llava_cfg = LlavaQwenConfig.from_pretrained(model_path)
                    print(f"Overwriting config with {overwrite_config}")
                    for k, v in overwrite_config.items():
                        setattr(llava_cfg, k, v)
                    model = LlavaQwenForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, config=llava_cfg, **kwargs)
                else:
                    model = LlavaQwenForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, **kwargs)

            elif "grok" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                    )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'llava' in model_name.lower() or 'grok' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        oct_2d_tower = model.get_oct_2d_tower()
        if not oct_2d_tower.is_loaded:
            oct_2d_tower.load_model(device_map=device_map)
        if device_map != 'auto':
            oct_2d_tower.to(device=device_map, dtype=torch.float16)
        else:
            oct_2d_tower.to(device='cuda', dtype=torch.float16)
        print(f"oct tower位置：{next(oct_2d_tower.parameters()).device}")

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if device_map != 'auto':
            vision_tower.to(device=device_map, dtype=torch.float16)
        else:
            vision_tower.to(device='cuda', dtype=torch.float16)
        image_processor = vision_tower.image_processor
        print(f"cfp tower位置：{next(vision_tower.parameters()).device}")

        mm_projector = model.get_model().mm_projector
        if mm_projector is not None:
            if device_map != 'auto':
                mm_projector.to(device=device, dtype=torch.float16)
            else:
                # 如果使用 auto device_map，确保 projector 在 cuda
                mm_projector.to(device='cuda', dtype=torch.float16)
        print(next(mm_projector.parameters()).device)

        oct_2d_projector = model.get_model().oct_2d_projector
        if oct_2d_projector is not None:
            if device_map != 'auto':
                oct_2d_projector.to(device=device, dtype=torch.float16)
            else:
                # 如果使用 auto device_map，确保 projector 在 cuda
                oct_2d_projector.to(device='cuda', dtype=torch.float16)
        print(next(oct_2d_projector.parameters()).device)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
