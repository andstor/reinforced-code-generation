# reinforced-code-generation


Setup accelerate:
```bash
accelerate config
```

Train with TRL:
```bash
accelerate launch run_trl.py \
--do_train \
--dataset_name openai_humaneval \
--dataset_config_name openai_humaneval \
--text_column_name prompt \
--model_name_or_path gpt2-xl \
--num_train_epochs 2 \
--per_device_train_batch_size 64 \
--forward_batch_size 16 \
--output_dir output \
--seed 42 \
--preprocessing_num_workers 8 \
--max_new_tokens 512 \
--report_to wandb
```

Train with TRL:
```bash
accelerate launch run_trl.py \
--do_train \
--dataset_name code_x_glue_tc_text_to_code \
--dataset_config_name default \
--text_column_name code \
--split_samples \
--model_name_or_path gpt2-xl \
--num_train_epochs 2 \
--per_device_train_batch_size 64 \
--forward_batch_size 1 \
--output_dir output \
--seed 42 \
--preprocessing_num_workers 8 \
--max_new_tokens 512 \
--report_to wandb
```


Train with TRL-X:
```bash
accelerate launch run_trlx.py \
--do_train \
--max_train_samples 100 \
--do_eval \
--max_eval_samples 100 \
--dataset_name code_search_net \
--text_column_name python \
--output_dir output \
--seed 42
```

Add Antlr4 to Java classpath:
```bash
export CLASSPATH=".:/opt/homebrew/Cellar/antlr/4.11.1/antlr-4.11.1-complete.jar:$CLASSPATH"
```

Generate Python3 and Java parser and lexer for the Python3 grammar:
```bash
make -f python3.mk all
````

Show parse tree in a GUI:
```bash
make -f python3.mk grun < test.py 
```