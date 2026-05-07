# Reproducing the Results
To set up the environment for reproducing the results of this paper, please follow the instructions below:
```bash
conda create -n myenv python=3.12
conda activate myenv
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

Before running the error analysis script, download the pre-trained ViT models:
```bash
bash convert_vits.sh
```
Make sure to place the downloaded models in the appropriate directory as specified in the `error_analysis_vit.sh` script.

## Running the Error Analysis (ViT)
To reproduce the results of ViT in this paper, please run the following command:
```bash
bash error_analysis_vit.sh
```
You can modify the `dataset_id` and `model_ids` variables in the `error_analysis_vit.sh` script to analyze different datasets and models as needed.

## Running the Error Analysis (BERT)
To reproduce the results of BERT in this paper, please run the following command:
```bash
bash error_analysis_bert.sh sst2
```
You can replace `sst2` with other datasets such as `agnews` or `imdb` to analyze different datasets as needed.

## Running the Error Analysis (GPT-2)
To reproduce the results of GPT-2 in this paper, please run the following command:
```bash
bash error_analysis_gpt2.sh
```