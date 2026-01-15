# JIGSAW

This repository hosts the code and data for the paper: **JIGSAW**.

#### Requirements:
- **Environment**: `requirements.txt` (Please use Python 3.9 for this repository)

  ```shell
  pip install -r requirements.txt
  ```

  If you get a `pyarrow` error, try this
  ```shell
  pip install --upgrade datasets pyarrow
  ```


- **Large Language Models to Edit**: 

  You have three options to load LLMs for editing:

  1. Download the LLMs you want to edit and put them in `./hugging_cache/` 

  2. Specify the path to your existing LLMs in the configuration files, e.g.,  `./hparams/FT/llama-7b.yaml`:

     ```yaml
     model_name: "your/path/to/LLMs"
     ```

  3. Provide the model name in the configuration files and the program will automatically employ `from_pretrained` to load the model:

     ```yaml
     model_name: "meta-llama/Meta-Llama-3-8B-Instruct"
     ```

  4. Copy the `.env.example` file for your own `.env` file
    ```shell
      cp .env.example .env
    ```
    And fill it up.
<!-- - **Manual Adjustments**:Some sections of the code require manual configuration based on your specific setup. Please search the codebase globally for the string **"TO-DO"** and edit those lines accordingly before running experiments. -->

  
- **Datasets**: The data of ZsRE, COUNTERFACT, and WikiBigEdit are provided in `./data/`

- **LLM-As-A-Judge Configuration**: To use the LLM-As-A-Judge functionality, you must provide your OpenRouter API key via environment variables.
  1. Create a `.env` file in the root directory.

  2. Add your key as follows:
    ```shell
    API_KEY=your_open_router_key_here
    ```

<!-- # AlphaEdit FT
```shell
python alphaedit_ft.py --model llama3-8b --rewrite_module model.layers.{}.mlp.down_proj.weight --batch_size 32 --device 1 --save_model_dir alphaedit_ft_llama3_zsre3k_bs32_noEOS --data_type zsre --eval_num 30 --cache_sample_num 1000
``` -->

<!-- # AlphaEdit
```shell
python edit.py --editing_method AlphaEdit --hparams_dir ./hparams/AlphaEdit/llama3-8b.yaml --data_path ./data/zsre_mend_eval_3k.json --datatype zsre --ds_size 3000 --batch_edit True --device 0 --down_eval True --save_model_dir alphaedit_llama3_wiki3k_bs3k
``` -->


#### Training
- **SnapEdit**
  ```shell
  python run_snap.py --model llama3-8b --data_type wiki --cache_sample_num 10000 --energy_threshold 0.8 --batch_size 32 --wandb_project JIGSAW
  ```
- **SnapEdit Sequential**
  ```shell
  python run_snap.py --model llama3-8b --data_type wiki --cache_sample_num 100 --energy_threshold 0.8 --batch_size 32 --wandb_project JIGSAW --sequential_edit True --num_edits 500
  ```
  > Note: It is important to set `--num_edits` whenever `--sequential_edit` is enabled to define the edit batch size or sequence limit.
- **MEMIT**
  ```shell
  python edit.py --model llama3-8b --data_type wiki --editing_method MEMIT --batch_size 32 --eval_every 512 --batch_edit True --wandb_project JIGSAW
  ```
- **UltraEdit**
  ```shell
  python edit.py --model llama3-8b --data_type wiki --editing_method UltraEdit --batch_size 32 --eval_every 512 --batch_edit True --wandb_project JIGSAW
  ```
- **AlphaEdit**
  ```shell
  python edit.py --model llama3-8b --data_type wiki --editing_method AlphaEdit --batch_size 32 --eval_every 512 --batch_edit True --wandb_project JIGSAW
  ```
- **WISE**
  ```shell
  python edit.py --model llama3-8b --data_type wiki --editing_method WISE --batch_size 1 --eval_every 512 --batch_edit False --wandb_project JIGSAW
  ```
- **MEND**
  ```shell
  python edit.py --model llama3-8b --data_type wiki --editing_method MEND --batch_size 1 --eval_every 512 --batch_edit False --wandb_project JIGSAW
  ```
- **Loc-BF-FT**
  ```shell
  python locft-bf.py --model llama3-8b --data_type wiki --batch_size 32 --wandb_project JIGSAW
  ```
- **Adam-NSCL**
  ```shell
  python alphaedit_ft.py --model llama3-8b --data_type wiki --cache_sample_num 10000 --energy_threshold 0.5 --batch_size 32 --wandb_project JIGSAW
  ```
#### Evaluate
- **Base Model (Base Capabilities)**
  ```shell
  python run_base_benchmarks.py --edited_model_dir models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2/ --model_name llama3-8b --alg_name Base --data_type wiki --eval_num 200
  ```
- **Base Model (Base Capabilities) (Example: Resume)**
  ```shell
  python run_base_benchmarks.py --edited_model_dir models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2/ --model_name llama3-8b --alg_name Base --data_type wiki --eval_num 200 --wandb_run_id your_wandb_run_id
  ```
- **Base Model (Edited Capabilities)**
  ```shell
  python run_edited_benchmarks.py --edited_model_dir models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2/ --model_name llama3-8b --max_length 40 --context_type qa_inst --alg_name Base --data_type wiki --evaluation_criteria llm_judge --eval_num 3000 
  ```
- **UltraEdit**
  ```shell
  python run_benchmarks.py --edited_model_dir llama3-8b_UltraEdit_wiki --model_name llama3-8b --max_length 40 --context_type qa_inst --alg_name JIGSAW --data_type wiki --eval_num 30 --evaluation_criteria exact_match
  ```
- **JIGSAW**
  ```shell
  python run_benchmarks.py --edited_model_dir llama3-8b_JIGSAW_wiki_0.5 --model_name llama3-8b --max_length 40 --context_type qa_inst --alg_name JIGSAW --data_type wiki --eval_num 30 --evaluation_criteria exact_match
  ```
<!-- #### SafeEdit Training
- **Jigsaw**
  ```shell
  python run_snap.py --model llama3-8b --data_type safeedit_train --cache_sample_num 10000 --energy_threshold 0.8 --batch_size 32 --wandb_project JIGSAW
  ```
#### SafeEdit Eval Edit
```shell
  python run_edited_benchmarks.py --edited_model_dir llama3-8b_JIGSAW_safeedit_train_0.95/ --model_name llama3-8b --max_length 100 --context_type chat_temp --alg_name Base --data_type safeedit_test --evaluation_criteria llm_judge --eval_num 1350 
  ```
#### SafeEdit Eval Base
```shell
  python run_base_benchmarks.py --edited_model_dir llama3-8b_JIGSAW_safeedit_train_0.95/ --model_name llama3-8b --alg_name Base --data_type safeedit_test --eval_num 20
  ``` -->
