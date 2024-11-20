# GLM-4-Voice-SFT

SFT code for GLM-4-voice

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
