# CopyRight wenzheliu, 2024
import os
import random
import json
import torch
from transformers import AutoTokenizer
from argparse import ArgumentParser
from tqdm import tqdm
from transformers import WhisperFeatureExtractor, AutoTokenizer, AutoModel

from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from speech_tokenizer.utils import extract_speech_token


def interleaved_process(list_text, list_speech, num_text=13, num_speech=26):
    result = []
    len_t, len_s = len(list_text), len(list_speech)
    assert len_t <= int(num_speech/num_text)*len_s, "text token length must be larger than 2 * speech token length"
    index_t, index_s = 0, 0

    while index_t < len_t:
        result.extend(list_text[index_t:index_t + num_text])
        index_t += num_text

        result.extend(list_speech[index_s:index_s + num_speech])
        index_s += num_speech

    if index_s < len_s:
        result.extend(list_speech[index_s:])

    return result

def get_message(content, role, whisper_model, feature_extractor):
    assert role in ['user', 'assistant']
    if role == 'user':
        tokens = extract_speech_token(whisper_model, feature_extractor, [content])
        tokens = tokens[0]
        tokens = "".join([f"<|audio_{x}|>" for x in tokens]) 
        tokens = "<|begin_of_audio|>" + tokens + "<|end_of_audio|>"
        token_ids = tokenizer([tokens], return_tensors="pt")['input_ids'].tolist()[0][2:]       

    else:
        audio_tokens = extract_speech_token(whisper_model, feature_extractor, [content['audio_path']])
        audio_tokens = audio_tokens[0]
        audio_tokens = "".join([f"<|audio_{x}|>" for x in audio_tokens])

        text_token_ids = tokenizer([content['text']], return_tensors="pt")['input_ids'].tolist()[0][2:] # [gMask][sop]
        audio_token_ids = tokenizer([audio_tokens], return_tensors="pt")['input_ids'].tolist()[0][2:] # [gMask][sop]
        token_ids = interleaved_process(text_token_ids, audio_token_ids, 13, 26)

    return token_ids

def gen_messages(item, truns, system_prompt):
    messages = [
                {
                    "role": "system",
                    "content": f"{system_prompt}",
                }
     ]

    for i in range(turns):
        user_wav_path = item['data'][i]['user']['wav_path']
        agent_wav_path = item['data'][i]['agent']['wav_path']
        agent_text = item['data'][i]['agent']['text']
        messages.append(
                        {
                            "role": "user",
                            "content": f"{user_wav_path}",
                        }
        )
        messages.append(
                        {
                            "role": "assistant",
                            "content": {"text": f"{agent_text}",
                                        "audio_path": f"{agent_wav_path}"}
                        }
        )

    return messages


if __name__ == "__main__":
    parser = ArgumentParser(
    description="genearte features for Interleave-Voice")
    parser.add_argument(
        'input_json_file',
        help='input json file path of dataset',
        action='store')
    parser.add_argument(
        'out_json_file',
        help='output json file path of features',
        action='store')
    args = parser.parse_args()

    input_json_file = args.input_json_file
    out_json_file = args.out_json_file

    device = "cuda"
    whisper_model = WhisperVQEncoder.from_pretrained("THUDM/glm-4-voice-tokenizer").eval().to(device)
    feature_extractor = WhisperFeatureExtractor.from_pretrained("THUDM/glm-4-voice-tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-4-voice-9b", trust_remote_code=True)
    end_token_id = tokenizer.convert_tokens_to_ids('<|user|>')

    system_prompt = "User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. "
    sys_prompt_str = f"<|system|>\n{system_prompt}"
    add_generation_prompt = True

    with open(input_json_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(out_json_file, 'w') as f:
        for i in tqdm(range(len(lines))):
            item = json.loads(lines[i])
            if item['turns'] == 0: continue

            turns = random.randint(1, item['turns'])
            messages = gen_messages(item, turns, system_prompt)

            assert (messages[-1]['role'] == 'assistant' or len([message for message in messages if message['role'] == 'assistant']) == 0), "messages如果有assistant的话必须最后一个role是assistant"
            query_messages, answer_messages = messages[:-1], [messages[-1]]

            # histoty and query
            input_ids, labels = [], []
            tmp = tokenizer([sys_prompt_str], return_tensors="pt")['input_ids'].tolist()[0]
            input_ids.extend(tmp)
            labels.extend([-100] * len(tmp))
            for i, message in enumerate(query_messages):
                if message['role'] == 'user':
                    tmp = tokenizer(['<|user|>\n'], return_tensors="pt")['input_ids'].tolist()[0][2:]
                    input_ids.extend(tmp)
                    labels.extend([-100] * len(tmp))
                    tmp = get_message(message['content'], 'user', whisper_model, feature_extractor)
                    input_ids.extend(tmp)
                    labels.extend([-100] * len(tmp))
                elif message['role'] == 'assistant':
                    tmp = tokenizer(['<|assistant|>streaming_transcription\n'], return_tensors="pt")['input_ids'].tolist()[0][2:]
                    input_ids.extend(tmp)
                    labels.extend([-100] * len(tmp))
                    tmp = get_message(message['content'], 'assistant', whisper_model, feature_extractor)
                    input_ids.extend(tmp)
                    labels.extend(tmp)

                if i == len(query_messages)-1 and add_generation_prompt:
                    tmp = tokenizer(['<|assistant|>streaming_transcription\n'], return_tensors="pt")['input_ids'].tolist()[0][2:]
                    input_ids.extend(tmp)
                    labels.extend([-100] * len(tmp))

            tmp = get_message(answer_messages[0]['content'], 'assistant', whisper_model, feature_extractor)
            input_ids.extend(tmp)
            labels.extend(tmp)
            input_ids.append(end_token_id)
            labels.append(end_token_id)

            data_dict = {''input_ids': input_ids, 'labels': labels}

            f.write(json.dumps(data_dict) + '\n')

