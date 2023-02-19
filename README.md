# reinforced-code-generation


Setup accelerate:
```bash
accelerate config
```

Train with TRL:
```bash
accelerate launch run_trl.py \
--do_train \
--max_train_samples 100 \
--dataset_name openai_humaneval \
--dataset_config_name python \
--text_column_name func_documentation_string \
--model_name_or_path gpt2 \
--per_device_train_batch_size 8 \
--forward_batch_size 1 \
--output_dir output \
--seed 42 \
--preprocessing_num_workers 10 \
--max_new_tokens 16 \
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
--text_column_name prompt \
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