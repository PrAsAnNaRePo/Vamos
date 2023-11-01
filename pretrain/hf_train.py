from dataclasses import dataclass, field
import os
from typing import Optional
import torch
from torch.utils.data import DataLoader, Dataset
from hf_data import LLavaDataset
from hf_model import VamosConfig, Vamos
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import Trainer, deepspeed, CLIPProcessor, AutoTokenizer
from transformers.trainer_pt_utils import LabelSmoother
from accelerate.utils import DistributedType

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

@dataclass
class ModelArguments:
    llm_id: Optional[str] = field(default="NousResearch/Nous-Capybara-3B-V1.9")
    clip_id: Optional[str] = field(default="openai/clip-vit-large-patch14-336")
    projector_layers: Optional[int] = field(default=8)
    projector_heads: Optional[int] = field(default=16)


@dataclass
class DataArguments:
    dataset_id: str = field(
        default="coco", metadata={"help": "Path to the training data."}
    )
    max_len: int = field(
        default=55, metadata={"help": "Max length of the text."}
    )
    dataset_limit: int = field(
        default=25, metadata={"help": "Limit of the dataset samples."}
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)

class LLaVATrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs) 
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.eval_batch_size,
            num_workers=self.args.dataloader_num_workers,
            shuffle=False,
            pin_memory=True,
        )
     
def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    (
        model_args,
        data_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()

    if getattr(training_args, 'deepspeed', None):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    local_rank = training_args.local_rank

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1

    config = VamosConfig(
        **model_args.__dict__,
    )
    processor = CLIPProcessor.from_pretrained(config.clip_id)
    tokenizer = AutoTokenizer.from_pretrained(config.llm_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": ["<img>", "</img>"]})
    config.img_start_token = tokenizer.convert_tokens_to_ids("<img>")
    model = Vamos(config=config, pad_token_id=tokenizer.pad_token_id)
    model.llm.resize_token_embeddings(len(tokenizer))

    train_set = LLavaDataset(
        processor,
        tokenizer,
        **data_args.__dict__,
    )

    eval_set = LLavaDataset(
        processor,
        tokenizer,
        **data_args.__dict__,
        split='validation'
    )
    
    data_module = dict(train_dataset=train_set, eval_dataset=eval_set)
    trainer = LLaVATrainer(
        model=model, args=training_args, **data_module
    )
    trainer.train(resume_from_checkpoint=True if training_args.resume_from_checkpoint else None)
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()