import copy
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import transformers
import ujson as json
from PIL import Image
from torch.utils.data import Dataset
# If you get rid of AutoProcessor, the code dosen't work.
from transformers import AutoProcessor

from .params import DataArguments

IMAGE_TOKEN_INDEX = -200
IGNORE_INDEX = -100
LLaVA_IMAGE_TOKEN = "<image>"

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        padding=True,
    ):
        super(LazySupervisedDataset, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        # rank0_print("Formatting inputs...Skip in lazy mode")
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]

        processor = self.processor
        if "image" in sources:
            image_file = sources["image"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]

            images = []
           
            for image_file in image_files:
                if not os.path.exists(image_file):
                    image_file = os.path.join(image_folder, image_file)
                images.append(Image.open(image_file).convert("RGB"))

        else:
            images = None

        sources = copy.deepcopy(llava_to_openai(sources['conversations']))

        all_input_ids = [torch.tensor([1])] # bos token id
        all_labels = [torch.tensor([-100])] # ignore bos token
        all_pixel_values = []
        all_image_sizes = []


        # TODO: Need to fix from here. Add bos, eos, and pixel values.
        for idx, j in range(0, len(sources), 2):
            user_input = sources[j]['value']
            gpt_response = sources[j + 1]['value']

            user_input = processor.tokenizer.apply_chat_template(user_input, tokenize=False)
            gpt_response = processor.tokenizer.apply_chat_template(gpt_response, tokenize=False)

            if idx == 0:
                inputs = processor(user_input, images, return_tensors='pt')
                prompt_input_ids = inputs['input_ids']
                all_pixel_values.append(inputs['pixel_values'])
                all_image_sizes.append(inputs['image_sizes'])

            else:
                prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, return_tensors='pt')['input_ids']

            response_input_ids = processor.tokenizer(gpt_response, add_special_tokens=False, return_tensors='pt')['input_ids']

            input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
            labels = torch.cat(
                [
                    torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),  
                    response_input_ids.squeeze(0),
                ],
                dim=0,
            )

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        all_input_ids.append(torch.tensor([32000]))  # eos token id
        all_labels.append(torch.tensor([32000]))  # ignore eos token
        
        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)

        # Handling pixel values and image sizes as tensors
        if len(all_pixel_values) > 0:
            pixel_values = torch.cat(all_pixel_values, dim=0)
            image_sizes = torch.cat(all_image_sizes, dim=0)
        else:
            pixel_values = None
            image_sizes = None

        attention_mask = (input_ids > -1000000).to(torch.long)

        data_dict = dict(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        return data_dict

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]

        pixel_values = [instance["pixel_values"] for instance in instances]
        pixel_values = torch.stack(pixel_values, dim=0)
        image_sizes = [instance["image_sizes"] for instance in instances]
        image_sizes = torch.stack(image_sizes, dim=0)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        return batch
    

def replace_image_tokens(input_string, start_count=1):
    count = start_count

    if LLaVA_IMAGE_TOKEN not in input_string:
        return input_string, count

    while LLaVA_IMAGE_TOKEN in input_string:
        input_string = input_string.replace(LLaVA_IMAGE_TOKEN, f"<|image_{count}|>", 1)
        count += 1

    return input_string, count

def llava_to_openai(conversations):
    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    image_count = 1  # Initialize image count here
    for conversation in conversations:
        transformed_content, image_count = replace_image_tokens(conversation["value"], image_count)
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": transformed_content,
        }
        transformed_data.append(transformed_entry)

    return transformed_data

def make_supervised_data_module(processor, data_args):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = LazySupervisedDataset(
        data_path=data_args.data_path, processor=processor, data_args=data_args
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=processor.tokenizer)

    return dict(train_dataset=sft_dataset,
                eval_dataset=None,
                data_collator=data_collator)