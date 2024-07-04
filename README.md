# Fine-tuning Phi3-Vision

This repository contains a script for training the [Phi3-Vision model](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct).

## Update

- [2024/07/03] Added WebUI demo.
- [2024/06/27] 🔥Supports multi-image training and inference.
- [2024/06/27] Supports saving the model into safetensor.

## Table of Contents

- [Installation](#installation)
  - [Using `requirements.txt`](#using-requirementstxt)
  - [Using `environment.yaml`](#using-environmentyaml)
- [Model Download](#model-download)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
  - [Full Finetuning](#full-finetuning)
  - [Finetune with LoRA](#finetune-with-lora)
    - [Merge LoRA Weights](#merge-lora-weights)
- [Inference](#inference)
  - [CLI Inference](#cli-inference)
  - [WebUI Demo](#gradio-infernce-webui)

## Supported Features

- Deepspeed
- LoRA, QLoRA
- Flexible selection of LoRA target modules
- Full-finetuning
- Finetuning `img_projector` and `vision_tower`
- Disable/enable Flash Attention 2
- Multi-image training and inference

## Installation

Install the required packages using either `requirements.txt` or `environment.yml`.

### Using `requirements.txt`

```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

### Using `environment.yaml`

```bash
conda env create -f environment.yaml
conda activate phi3v
pip install flash-attn --no-build-isolation
```

**Note:** You should install the `flash-attn` after running other libraries with `requirements.txt` or `environment.yaml`.

## Model Download

Before training, download the Phi3-V model from HuggingFace. It is recommended to use the `huggingface-cli` to do this.

1. Install the HuggingFace CLI:

```bash
pip install -U "huggingface_hub[cli]"
```

2. Download the model:

```bash
huggingface-cli download microsoft/Phi-3-vision-128k-instruct --local-dir Phi-3-vision-128k-instruct --resume-download
```

## Dataset Preparation

The script requires a dataset formatted according to the LLaVA specification. The dataset should be a JSON file where each entry contains information about conversations and images. Ensure that the image paths in the dataset match the provided `--image_folder`.<br>

**When using multi-image dataset, the image tokens should all be `<image>`.**

<details>
<summary>Example Dataset</summary>

```json
[
  {
    "id": "000000033471",
    "image": "000000033471.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat are the colors of the bus in the image?"
      },
      {
        "from": "gpt",
        "value": "The bus in the image is white and red."
      },
      {
        "from": "human",
        "value": "What feature can be seen on the back of the bus?"
      },
      {
        "from": "gpt",
        "value": "The back of the bus features an advertisement."
      },
      {
        "from": "human",
        "value": "Is the bus driving down the street or pulled off to the side?"
      },
      {
        "from": "gpt",
        "value": "The bus is driving down the street, which is crowded with people and other vehicles."
      }
    ]
  }
  ...
]
```

</details>

## Training

To run the training script, use the following command:

### Full Finetuning

```bash
bash scripts/finetune.sh
```

### Finetune with LoRA

If you want to train with LoRA:

```bash
bash scripts/finetune_lora.sh
```

<details>
<summary>Training arguments</summary>

- `--deepspeed` (str): Path to DeepSpeed config file (default: "scripts/zero2.json").
- `--data_path` (str): Path to the LLaVA formatted training data (a JSON file). **(Required)**
- `--image_folder` (str): Path to the images folder as referenced in the LLaVA formatted training data. **(Required)**
- `--model_id` (str): Path to the Phi3-vision model. **(Required)**
- `--output_dir` (str): Output directory for model checkpoints (default: "output/test_train").
- `--num_train_epochs` (int): Number of training epochs (default: 1).
- `--per_device_train_batch_size` (int): Training batch size per GPU per forwarding step.
- `--gradient_accumulation_steps` (int): Gradient accumulation steps (default: 4).
- `--freeze_vision_tower` (bool): Option to freeze vision_model (default: False)
- `--tune_img_projector` (bool): Option to finetune img_projector (default: True)
- `--num_lora_modules` (int): Number of target modules to add LoRA (-1 means all layers).
- `--multimodal_lr` (float): Learning rate for multimodal modules (`vision_tower` and `img_projection`)
- `--learning_rate` (float): Learning rate for language module.
- `--bf16` (bool): Option for using bfloat16.
- `--lora_namespan_exclude` (str): Exclude modules with namespans to add LoRA.
- `--max_seq_length` (int): Maximum sequence length (defaut: 128K).
- `--bits` (int): Quantization bits (default: 16).
- `--disable_flash_attn2` (bool): Disable Flash Attention 2.
- `--report_to` (str): Reporting tool (choices: 'tensorboard', 'wandb', 'none') (default: 'tensorboard').
- `--logging_dir` (str): Logging directory (default: "./tf-logs").
- `--lora_rank` (int): LoRA rank (default: 128).
- `--lora_alpha` (int): LoRA alpha (default: 256).
- `--lora_dropout` (float): LoRA dropout (default: 0.05).
- `--logging_steps` (int): Logging steps (default: 1).
- `--dataloader_num_workers` (int): Number of data loader workers (default: 4).

**Note:** The learning rate of `vision_model` should be 10x ~ 5x smaller than the `language_model`.

</details>

#### Merge LoRA Weights

```
bash scripts/merge_lora.sh
```

**Note:** Remember to replace the paths in `finetune.sh` or `finetune_lora.sh` with your specific paths. (Also in `merge_lora.sh` when using LoRA.)

#### Issue for libcudnn error

```
Could not load library libcudnn_cnn_train.so.8. Error: /usr/local/cuda-12.1/lib/libcudnn_cnn_train.so.8: undefined symbol: _ZN5cudnn3cnn34layerNormFwd_execute_internal_implERKNS_7backend11VariantPackEP11CUstream_stRNS0_18LayerNormFwdParamsERKNS1_20NormForwardOperationEmb, version libcudnn_cnn_infer.so.8
```

You could run `unset LD_LIBRARY_PATH` for this error.
You could see this [issue](https://github.com/andimarafioti/florence2-finetuning/issues/2)

## Inference

**Note:** You should use the merged weight when trained with LoRA.

### CLI Inference

```
python -m src.serve.cli \
 --model-path /path/to/merged/weight \
 --image-file /Path/to/image
```

You can set some other generation configs like `repetition_penalty`, `temperature` etc.

### Gradio Infernce (WebUI)

1. Install gradio

```
pip install gradio
```

2. Launch app

```
python -m src.serve.app \
    --model-path /path/to/merged/weight
```

You can launch gradio based demo with this command. This can also set some other generation configs like `repetition_penalty`, `temperature` etc.

## TODO

- [x] Saving in safetensor
- [x] Supporting multi-image training and inference.
- [x] Demo with WebUI
- [ ] Converting into gguf format.

## Known Issues

- [libcudnn issue](#issue-for-libcudnn-error)
- Does not support text-only data.

## License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details.

## Citation

If you find this repository useful in your project, please consider giving a :star: and citing:

```bibtex
@misc{phi3vfinetuning2023,
  author = {Gai Zhenbiao and Shao Zhenwei},
  title = {Phi3V-Finetuning},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/GaiZhenbiao/Phi3V-Finetuning},
  note = {GitHub repository},
}

@misc{phi3-vision-ft,
  author = {Yuwon Lee},
  title = {Phi-3-vision-ft},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/2U1/Phi3-Vision-ft},
  note = {GitHub repository, forked and developed from \cite{phi3vfinetuning2023}},
}
```

## Acknowledgement

This project is based on

- [LLaVA](https://github.com/haotian-liu/LLaVA): An amazing open-source project of LMM.
- [Mipha](https://github.com/zhuyiche/llava-phi): Open-source projcet of SMM with amazing capabilites.
- [Microsoft Phi-3-vision-128k-instruct](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct): Awesome pretrained SMM using phi3.
- [Phi3V-Finetuning](https://github.com/GaiZhenbiao/Phi3V-Finetuning): Open-source project for finetuning phi-3-vision.
