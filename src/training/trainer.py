import os
import torch
import torch.nn as nn

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    ALL_LAYERNORM_LAYERS,
    is_peft_available,
    WEIGHTS_NAME,
    TRAINING_ARGS_NAME,
    SAFE_WEIGHTS_NAME,
    logger,
)
import safetensors
from peft import PeftModel
from typing import Optional
from transformers.processing_utils import ProcessorMixin
from transformers.modeling_utils import PreTrainedModel
from peft import PeftModel

class Phi3VTrainer(Trainer):

    def __init__(self, *args, processor: Optional[ProcessorMixin] = None, **kwargs):
        super(Phi3VTrainer, self).__init__(*args, **kwargs)
        self.processor = processor

    # This code is from LLaVA
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.vision_lr is not None and self.args.projector_lr is not None:
                
                vision_parameters = [
                    name for name, _ in opt_model.named_parameters() 
                    if "vision_model" in name 
                ]
                
                # glb_GN and sub_GN are the parameters for merging the output of channel dimension
                img_projection_parameters = [
                    name for name, _ in opt_model.named_parameters() 
                    if "img_projection" in name or "glb_GN" in name or "sub_GN" in name
                ]

                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in vision_parameters and n not in img_projection_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in vision_parameters and n not in img_projection_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in vision_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.vision_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in vision_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.vision_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in img_projection_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in img_projection_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer


    def _save_checkpoint(self, model, trial, metrics=None):
        super(Phi3VTrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
            # If we are executing this function, we are the process zero, so we don't check for that.
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Saving model checkpoint to {output_dir}")

            supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
            # Save a trained model and configuration using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            if not isinstance(self.model, supported_classes):
                if state_dict is None:
                    state_dict = self.model.state_dict()

                if isinstance(self.accelerator.unwrap_model(self.model), supported_classes):
                    self.accelerator.unwrap_model(self.model).save_pretrained(
                        output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                    )
                else:
                    logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                    if self.args.save_safetensors:
                        safetensors.torch.save_file(
                            state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                        )
                    else:
                        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
            else:
                state_dict = {k:v for k, v in state_dict.items() if "wte" not in k}
                self.model.save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )

            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)

            if self.processor is not None:
                self.processor.save_pretrained(output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    # def training_step(self, model, inputs):
    #     for name, param in model.named_parameters():
    #         if 'vision_model' in name and param.requires_grad:
    #             print(f"Training parameter {name}")
            
    #         elif 'img_projection' in name and param.requires_grad:
    #             print(f"Training parameter {name}")
    #     return super().training_step(model, inputs)