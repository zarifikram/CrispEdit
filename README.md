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
- **Jigsaw**
  ```shell
  python jigsaw.py --model llama3-8b --data_type wiki --cache_sample_num 1000 --energy_threshold 0.5 --batch_size 32 --wandb_project JIGSAW
  ```

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
- **Loc-BF-FT**
  ```shell
  python locft-bf.py --model llama3-8b --data_type wiki --batch_size 32 --wandb_project JIGSAW
  ```
- **AlphaEdit FT**
  ```shell
  python alphaedit_ft.py --model llama3-8b --data_type wiki --cache_sample_num 10000 --energy_threshold 0.5 --batch_size 32 --wandb_project JIGSAW
  ```

#### Evaluate
  ```shell
  python evaluate.py --edited_model_dir llama3-8b_JIGSAW_wiki_0.5 --model_name llama3-8b --max_length 40 --context_type qa_inst --alg_name JIGSAW --data_type wiki --eval_num 30 --evaluation_criteria exact_match
  ```
