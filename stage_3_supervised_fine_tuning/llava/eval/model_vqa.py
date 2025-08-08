import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torchvision import transforms
import numpy as np

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def load_oct_sampled(oct_dir, num_sample_frames=8):
    """
    加载并采样OCT文件夹中的帧。
    注意：为了评估的确定性，这里默认采样固定数量的帧，而不是随机数量。
    """
    oct_files = sorted([f for f in os.listdir(oct_dir) if f.endswith('.png') or f.endswith('.jpg')])

    # 在评估时，通常我们采样固定的帧数以保证结果可复现
    if len(oct_files) < num_sample_frames:
        # 如果帧数不够，就重复使用现有帧
        indices = np.arange(len(oct_files))
        indices = np.tile(indices, (num_sample_frames + len(indices) - 1) // len(indices))
        idxs = indices[:num_sample_frames]
    elif num_sample_frames == 1:
        idxs = [4]
    else:
        idxs = np.linspace(0, len(oct_files) - 1, num=num_sample_frames, dtype=int)

    # 定义与训练时一致的预处理器
    oct_processor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    octs = []
    for idx in idxs:
        image_path = os.path.join(oct_dir, oct_files[idx])
        print(image_path)
        oct_frame = Image.open(image_path).convert('RGB')
        octs.append(oct_processor(oct_frame))

    return torch.stack(octs, dim=0)  # [num_sample_frames, C, H, W]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        # 1. CFP 加载
        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]

        # 2. 加载 OCT
        octs_tensor = None
        if 'octs' in line and args.oct_folder:
            print('oct能找到----------------')
            oct_dir = os.path.join(args.oct_folder, line["octs"])
            octs_tensor = load_oct_sampled(oct_dir, num_sample_frames=args.num_oct_frames)
        else:
            print("没找到oct---------------")

        with torch.inference_mode():
            # 在generate前添加检查
            # print(f"Input IDs shape: {input_ids.shape}")
            # print(f"Image tensor range: {image_tensor.min():.3f} to {image_tensor.max():.3f}")
            if octs_tensor is not None:
                print(f"OCT tensor range: {octs_tensor.min():.3f} to {octs_tensor.max():.3f}")
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                octs=octs_tensor.unsqueeze(0).half().cuda() if octs_tensor is not None else None,  # 传入OCT张量
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--oct-folder", type=str, default=None)
    parser.add_argument("--num-oct-frames", type=int, default=1)
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
