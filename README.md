# ChatGLMVoice-SFT

a simple SFT project for GLM-4-voice

## feature extraction
input.json is a file contains several multi-turn QA, such as 
```
{'id': 0, 'turns': 2, 'data': [{"user": {"wav_path": "q1.wav"}, "agent": {"wav_path": "a1.wav", "text": "xxxx",}}, {"user": {"wav_path": "q2.wav"}, "agent": {"wav_path": "a2.wav", "text": "xxxx",}}],}\n
  ...
```

```bash
python feature_extractor/feature_extractor.py --input_json_file input.json --out_json_file data/data.json
```

## train

```bash
# choose lora or all sft
sh run.sh
```

## inference

```bash
python merge_lora.py --model_dir out/epoch_1_step_500 --merge_dir out/epoch_1_step_500
```
then call inference function in [GLM-4-Voice](https://github.com/THUDM/GLM-4-Voice/)

## Acknowledgements
Some code in this project is from:

[GLM-4-Voice](https://github.com/THUDM/GLM-4-Voice/)

[ChatGLM-Finetuning](https://github.com/liucongg/ChatGLM-Finetuning)
