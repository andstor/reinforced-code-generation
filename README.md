# reinforced-code-generation


Setup accelerate:
```bash
accelerate config
```

Run the script:
```bash
accelerate launch run_trl.py \
--do_train \
--max_train_samples 100 \
--dataset_name code_search_net \
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


```bash
accelerate launch run_trlx.py \
--do_train \
--max_train_samples 100 \
--do_eval \
--max_eval_samples 100 \
--dataset_name code_search_net \
--dataset_config_name python \
--text_column_name func_documentation_string \
--output_dir output \
--seed 42
```