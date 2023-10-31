# Controlled Text Generation with Hidden Representation Transformations
This is the code repository for our paper submitted to ACL 2023: [Controlled Text Generation with Hidden Representation Transformations](todo)

# Abtract
We propose CHRT (Control Hidden Representation Transformation) -- a controlled language generation framework that steers large language models to generate text pertaining to certain attributes (such as toxicity). CHRT gains attribute control by modifying the hidden representation of the base model through learned transformations. We employ a contrastive-learning framework to learn these transformations that can be combined to gain multi-attribute control. The effectiveness of CHRT is experimentally shown by comparing it with seven existing baselines over three attributes. CHRT outperforms all the baselines in the task of detoxification, positive sentiment steering, and text simplification while minimizing the loss in linguistic qualities. Further, our approach has the lowest inference latency of only 0.01 seconds more than the base model, making it the most suitable for high-performance production environments. We open-source our code and release two novel datasets to further propel controlled language generation research.

# Dependencies
The authors run the code in following envirionment:
- Python 3.7+
- Conda Environment: `conda env create -f environment.yml`

# Downloads
To get CHRT models for toxicity, sentiment and simplicity simply run `download.sh`. 
  
# Generation
```
python3 generate.py \
	--prompt_path ../data/real_pwkp_prompts/prompts.txt \
    --model_type gpt2 \
    --base_model_type gpt2-medium \
    --chrt_model_path ../models/pwkp/best_model.pt \
    --generations_output_dir ../ \
    --generations_output_fname out.jsonl \
    --batch_size 128 \
    --n_gens 25 \
    --n_tokens_gen 25 \
    --verbose
```
`prompt_path` can be a new-line seperated text file of prompts or a JSONL file or a csv with prompts in the column with name `prompt_text`. 


# Training
First fine-tune the positive and negative guider model as follows:
```
#Fine-tune Positive Model
python3 -m finetune_gpt2 \
	--output_dir models/positive \
	--model_type gpt2-medium \
	--model_name_or_path gpt2-medium \
	--do_train \
	--num_train_epochs 3 \
	--block_size 128 \
	--save_total_limit 1 \
	--dataloader_drop_last \
	--per_device_train_batch_size 4 \
	--gradient_accumulation_steps 16 \
	--train_data_file ../data/real_sentiment_prompts/positive_sentiment_text.txt \
	--overwrite_cache\
	--overwrite_output_dir

#Fine-tune Negative Model
python3 -m finetune_gpt2 \
	--output_dir models/negative \
	--model_type gpt2-medium \
	--model_name_or_path gpt2-medium \
	--do_train \
	--num_train_epochs 3 \
	--block_size 128 \
	--save_total_limit 1 \
	--dataloader_drop_last \
	--per_device_train_batch_size 4 \
	--gradient_accumulation_steps 16 \
	--train_data_file ../data/real_sentiment_prompts/negative_sentiment_text.txt \
	--overwrite_cache\
	--overwrite_output_dir    
```
`negative_corpus.txt` and `positive_corpus.txt` contain negative and positive attribute text respectively.

Now train the CHRT head as follows:
```
python3 train.py \
	--task sentiment \
	--train_path ../data/real_sentiment_prompts/train.csv \
	--test_path ../data/real_sentiment_prompts/test.csv \
    --model_type gpt2 \
    --model_path_or_name gpt2-medium \
    --positive_model_path models/sentiment/positive_medium \
    --negative_nodel_path models/sentiment/negative_medium \
    --model_output_dir models/CHRT \
    --batch_size 16 \
    --n_epochs 2 \
    --triplet_weight 1 \
    --l2_weight 2
```


# Citation
Kumar, Vaibhav, Hana Koorehdavoudi, Masud Moshtaghi, Amita Misra, Ankit Chadha, and Emilio Ferrara. "Controlled text generation with hidden representation transformations." (2023).