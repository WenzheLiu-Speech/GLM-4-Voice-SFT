# -*- coding:utf-8 -*-
# CopyRight 刘聪NLP, 2023
# CopyRight wenzheliu, 2024

import torch
from transformers import AutoTokenizer, AutoModel
from model.modeling_chatglm import ChatGLMForConditionalGeneration
import argparse
from peft import PeftModel


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default="", type=str)
    parser.add_argument('--merge_dir', required=True, type=str)

    return parser.parse_args()


if __name__ == '__main__':
    args = set_args()
    base_model = AutoModel.from_pretrained("THUDM/glm-4-voice-9b", trust_remote_code=True)
    lora_model = PeftModel.from_pretrained(base_model, args.model_dir) # , torch_dtype=torch.float16)
    lora_model.to("cpu")
    model = lora_model.merge_and_unload()
    ChatGLMForConditionalGeneration.save_pretrained(model, args.merge_dir, max_shard_size="10GB")
