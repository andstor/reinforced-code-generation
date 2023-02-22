#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    is_torch_tpu_available,
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, set_seed
import evaluate



# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.26.0.dev0")

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--text_column_name",
        type=str,
        default=None,
        help="The column name of the dataset to use.",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--forward_batch_size",
        type=int,
        default=8,
        help="Batch size for the forward pass.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="Whether to run training.",
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="Whether to run eval on the dev set.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="The maximum number of new tokens to generate."
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of training examples to this value if set.",
    )
    parser.add_argument(
        "--split_examples",
        action="store_true",
        help="Whether to only use part of examples for training and evaluation.",
    )



    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, json or txt file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, json or txt file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main():
    args = parse_args()

    # Acellerator class for only pre-setups
    accelerator = Accelerator()


    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    #    #if "validation" not in raw_datasets.keys():
    #    #    raw_datasets["validation"] = load_dataset(
    #    #        args.dataset_name,
    #    #        args.dataset_config_name,
    #    #        split=f"train[:{args.validation_split_percentage}%]",
    #    #    )
    #    #    raw_datasets["train"] = load_dataset(
    #    #        args.dataset_name,
    #    #        args.dataset_config_name,
    #    #        split=f"train[{args.validation_split_percentage}%:]",
    #    #    )
    #else:
    #    data_files = {}
    #    dataset_args = {}
    #    if args.train_file is not None:
    #        data_files["train"] = args.train_file
    #    if args.validation_file is not None:
    #        data_files["validation"] = args.validation_file
    #    extension = args.train_file.split(".")[-1]
    #    if extension == "txt":
    #        extension = "text"
    #        dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
    #    raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
    #    # If no validation data is there, validation_split_percentage will be used to divide the dataset.
    #    if "validation" not in raw_datasets.keys():
    #        raw_datasets["validation"] = load_dataset(
    #            extension,
    #            data_files=data_files,
    #            split=f"train[:{args.validation_split_percentage}%]",
    #            **dataset_args,
    #        )
    #        raw_datasets["train"] = load_dataset(
    #            extension,
    #            data_files=data_files,
    #            split=f"train[{args.validation_split_percentage}%:]",
    #            **dataset_args,
    #        )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer, padding_side='left') #TODO: check if this is correct
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer, padding_side='left') #TODO: check if this is correct
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    tokenizer.pad_token = tokenizer.eos_token

    
    if args.model_name_or_path:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        model = AutoModelForCausalLMWithValueHead.from_config(config)
        ref_model = AutoModelForCausalLMWithValueHead.from_config(config)
        n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.pretrained_model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if args.do_train: #TODO: add argument do_train
        column_names = list(raw_datasets["train"].features) #TODO: change to train
    else:
        column_names = list(raw_datasets["validation"].features)
    if args.text_column_name is not None:
        text_column_name = args.text_column_name
    else:
        text_column_name = "text" if "text" in column_names else column_names[0]
        logger.warning(f"Using column {text_column_name} as text column.")


    max_input_length = model.pretrained_model.config.max_position_embeddings - args.max_new_tokens

    # Tokenize the data
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], max_length=max_input_length, truncation=True)

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    if args.do_train: #TODO: add argument do_train
        if "train" not in tokenized_datasets: #TODO: change to train
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"] #TODO: change to train
        if args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    """
    if args.do_eval: #TODO: add argument do_eval
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if args.max_eval_samples is not None: #TODO: add argument max_eval_samples
            max_eval_samples = min(len(eval_dataset), args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)
    """

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

   
    train_dataset = tokenized_datasets["train"] #TODO: change to train
    #train_dataset.set_format(type="torch")

    #eval_dataset = tokenized_datasets["validation"]
    #eval_dataset.set_format(type="torch")

    # collate function
    def data_collator(examples):
        # first = examples[0]
        batch = {}
        # for col in first.keys():
        #    batch[col] = [d[col] for d in examples]
        #batch = tokenizer.pad(examples)
        batch["input_ids"] = [torch.tensor(ids["input_ids"]) for ids in examples]
        #batch["attention_mask"] = [torch.tensor(ids["attention_mask"]) for ids in examples]
        #batch["input_ids"] = torch.tensor(batch["input_ids"])
        #batch["attention_mask"] = torch.tensor(batch["attention_mask"])
        return batch
    

    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir

    ppo_config = PPOConfig(
        learning_rate=args.learning_rate, #1.41e-5, 
        log_with=args.report_to,
        accelerator_kwargs=accelerator_log_kwargs,
        batch_size=args.per_device_train_batch_size,
        forward_batch_size=args.forward_batch_size
    )

    #Initialize the PPOTrainer
    trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=train_dataset if args.do_train else None,
        #train_dataset=train_dataset if training_args.do_train else None, # TODO: add train dataset functionality
        #eval_dataset=eval_dataset if training_args.do_eval else None, # TODO: add eval dataset functionality
        data_collator=data_collator,
    )

    logger.info(trainer.accelerator.state, main_process_only=False)
    if trainer.accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()



    generation_kwargs = {
        "min_length":-1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": args.max_new_tokens,
        #TODO: add early stopping based on brace matching
    }

    from reward_fn import RewardFn
    rewardfn = RewardFn()

    """
    # Calibrate the reward function
    samples = train_dataset[:100]
    encodings = tokenizer.pad(samples)
    for k, v in encodings.items():
        # convert to tensors
        if isinstance(v, list):
            encodings[k] = torch.tensor(v)

    with torch.no_grad():
        gen_tokens = model.generate(
            **encodings,
            **generation_kwargs,
        )
        prompts = tokenizer.batch_decode(encodings.input_ids, skip_special_tokens=True)
        outputs = [tokenizer.decode(gen_tokens[i, encodings.input_ids.shape[1]:], skip_special_tokens=True) for i in range(gen_tokens.shape[0])]

    rewardfn.calibrate(prompts, outputs)
    """
    rewardfn.avg_tokens =170.38
    rewardfn.avg_errors = 50.39

    # Setup the reward function
    # This is a generator that yields a random number between 0 and 1
    # def reward_fn(samples, prompts, outputs):
        # """
        # This function is called by the trainer to calculate the reward for each sample.
        # The reward is a float between 0 and 1.

        # Args:

            # samples: The samples that were generated by the model. Prompts are prepended to the outputs.
            # prompts: The prompts that were used to generate the samples.
            # outputs: The outputs of the model. This is the same as samples, but without the prompts.
        # """
        # return random.uniform(0,1) # TODO: replace with actual reward function
        # # TODO: convert to tenso
    #

    

    #See https://github.com/CarperAI/trlx/blob/b91da7b03d8e9fa0c0d6dce10a8f2611aca3013f/trlx/trainer/accelerate_base_trainer.py#L201
    # for how to create a stopping condition
    # https://github.com/CarperAI/trlx/pull/172
    # And https://discuss.huggingface.co/t/stopping-criteria-for-batch/26564
    

    #trainer = trlx.train('gpt2', reward_fn=lambda samples, **kwargs: [sample.count('cats') for sample in samples])

    # Only show the progress bar once on each machine.
    pbar = tqdm(range(args.num_train_epochs*len(train_dataset)), disable=not trainer.accelerator.is_local_main_process, position=0)
    #completed_steps = 0
    #starting_epoch = 0
    for epoch in range(args.num_train_epochs):
        for _, batch in enumerate(trainer.dataloader):
            query_tensors = batch['input_ids']

            #### Get response from gpt2
            response_tensors = []
            prompt_tensors = []
            #num_rollouts = len(query_tensors)
            #pbar.set_description(f"[rollout 0 / {num_rollouts}]")
            #rollouts = 0
            #subbar = tqdm(total=num_rollouts, leave=False, position=accelerator.state.process_index+1)
            for query in query_tensors:
                # args.split_examples
                gen_len = args.max_new_tokens #output_length_sampler()
                generation_kwargs["max_new_tokens"] = gen_len

                #print("GENERATING...")
                
                #subbar.set_description(f"Generating {gen_len} tokens")
                if args.split_examples:
                    split_index = len(query)//2 #output_length_sampler()
                    prompt = query[:-split_index]
                else:
                    prompt = query
                prompt_tensors.append(prompt)
                response = trainer.generate(prompt, **generation_kwargs)
                response_tensors.append(response.squeeze()[-gen_len:])
                #rollouts += 1
                #subbar.set_description(f"[rollout {rollouts} / {num_rollouts}]")
            batch['response'] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
            batch['query'] = [tokenizer.decode(p.squeeze()) for p in prompt_tensors]
            batch['sample'] = [tokenizer.decode(s.squeeze()) for s in query_tensors]
            
            #### Compute reward
            # texts = batch['response']
            # rewards = [reward_fn() for _ in texts]
            #rewards = [torch.tensor(rewardfn(x)) for x in batch['response']]
            if args.split_examples:
                rewards = [torch.tensor(rewardfn(batch['query'][i] + batch['response'][i])) for i in range(len(batch['response']))]
            else:
                rewards = [torch.tensor(rewardfn(r)) for r in batch['response']]

            #### Run PPO step 
            stats = trainer.step(prompt_tensors, response_tensors, rewards)
            pbar.update(args.per_device_train_batch_size)
            trainer.log_stats(stats, batch, rewards)

        pbar.close()

    if args.with_tracking:
        trainer.accelerator.end_training()

    if args.output_dir is not None:
        trainer.accelerator.wait_for_everyone()
        trainer.save_pretrained(args.output_dir)
        if args.push_to_hub:
            repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

            #with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            #    json.dump({"perplexity": perplexity}, f)


if __name__ == "__main__":
    main()