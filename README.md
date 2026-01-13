# SanHinCorp1.0: A Benchmark Parallel Corpus for Low-Resource Sanskrit-Hindi Machine Translation

## Transformer Model

### 1. Create Conda Environment
```bash
mkdir SanHinCorp1.0
cd SanHinCorp1.0
conda env create -f  Transformer_environment.yml
conda activate transformer
mkdir Transformer
cd Transformer

```
### 2. Data Preprocessing


```bash
mkdir data
cd data
bash split_dataset.sh corpus.clean.bpe.32000.sa corpus.clean.bpe.32000.hi

bash data_preparation.sh ../data/train/train.sa ../data/train/train.hi

fairseq-preprocess --source-lang sa --target-lang hi --trainpref train/train --validpref dev/dev --testpref test/test --destdir tokenized.sa-hi --thresholdsrc 2 --thresholdtgt 2
```
### 2. Training
```bash
fairseq-train /home/kpdubey/nmt/nmt-sanhin-opusdata2/data/tokenized.sa-hi --arch transformer --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 --encoder-embed-dim 256 --encoder-ffn-embed-dim 512 --encoder-layers 3 --encoder-attention-heads 8 --encoder-learned-pos --decoder-ffn-embed-dim 512 --decoder-layers 3 --decoder-attention-heads 8 --decoder-learned-pos --max-epoch 25 --optimizer adam --lr 5e-4 --batch-size 128 --seed 1 --save-dir ../TransformerModel2 --fp16 --distributed-world-size 1 --keep-last-epochs 3 --skip-invalid-size-inputs-valid-test --tensorboard-logdir ../tensorboard_logs --criterion label_smoothed_cross_entropy --label-smoothing 0.1

```

### 3. Inference
```bash
fairseq-generate /home/kpdubey/nmt/nmt-sanhin-mydata/data/tokenized.sa-hi   --path ../TransformerModel2/checkpoint_best.pt   --batch-size 64   --beam 5   --remove-bpe > generated_translations2.txt

grep ^H generated_translations2.txt | cut -f3- > sys.out
grep ^T generated_translations2.txt | cut -f2- > ref.out
grep ^S generated_translations2.txt | cut -f2- > src.out
```
### 4. Evaluation: Compute BLEU4, chrF2, TER, COMET, METEOR, BERTScore Scores

```bash
python BLEU_chrF_TER.py
python COMET_METEOR.py
python BERTscore.py
```

## IndicTrans2 Model

### 1. Create Conda Environment
```bash
mkdir SanHinCorp1.0
cd SanHinCorp1.0
conda env create -f  IndicTrans2_environment.yml
conda activate IndicTrans2
mkdir it2
cd it2

```

### 2. Create Experiment Folder
```bash
mkdir indic-indic-exp
cd indic-indic-exp
mkdir train devtest vocab final_bin
```

#### Folder Structure
```
indic-indic-exp
├── train
│   ├── san_Deva-hin_Deva
│       ├── train.san_Deva
│       └── train.hin_Deva
│
├── devtest
│   ├── all
│   ├── san_Deva-hin_Deva
│       ├── dev.san_Deva
│       ├── dev.hin_Deva
│       ├── test.san_Deva
│       └── test.hin_Deva
│
├── vocab
│   ├── model.SRC
│   ├── model.TGT
│   ├── vocab.SRC
│   └── vocab.TGT
└── final_bin
    ├── dict.SRC.txt
    └── dict.TGT.txt
```

### 3. Clone IndicTrans2 Repository 
```bash
git clone https://github.com/AI4Bharat/IndicTrans2
```

### 4. Install Dependencies
```bash

source install.sh
```


### 5. Data Preparation & Binarization
```bash
cd IndicTrans2
bash prepare_data_joint_finetuning.sh ../indic-indic-exp
```

### 6. Finetuning
download the model
https://huggingface.co/ai4bharat/indictrans2-indic-indic-1B
```bash

bash finetune.sh ../indic-indic-exp transformer_s /SanHinCorp1.0/it2/indic-indic-exp/model/checkpoint_best.pt
```

### 7. Inference
```bash
bash joint_translate.sh <infname> <outfname> <src_lang> <tgt_lang> <ckpt_dir>
bash joint_translate.sh ../indic-indic-exp/devtest/all/san_Deva-hin_Deva/test.san_Deva san2hin_output.txt san_Deva hin_Deva ../indic-indic-exp/
```

### 8. Evaluation: Compute BLEU4, chrF2, TER, COMET, METEOR, BERTScore Scores
```bash
cd ..
cd output
python BLEU_chrF_TER.py
python COMET_METEOR.py
python BERTscore.py
```

## mBART50 Model

### 1. Create Environment and Project Structure
```bash
mkdir SanHinCorp1.0
cd SanHinCorp1.0
conda env create -f  mBART50_environment.yml
conda activate mBART50
mkdir mBART50
cd mBART50
```

### 2. Download mBART50 Model from HuggingFace
Ensure that all dataset files are in `.json` format (e.g., `train.json`, `valid.json`, `test.json`).

### 3. Training
```bash
python3 transformers/examples/pytorch/translation/run_translation.py \
    --model_name_or_path facebook/mbart-large-50-many-to-many-mmt \
    --do_train --do_eval \
    --source_lang ma_XX --target_lang hi_IN \
    --train_file /home/SanHinCorp1.0/mBART50/data/train/train.json \
    --validation_file /home/SanHinCorp1.0/mBART50/data/dev/dev.json \
    --test_file /home/SanHinCorp1.0/mBART50/data/test/test.json \
    --output_dir checkpoint/mBART50_san_hin/ \
    --per_device_train_batch_size=6 \
    --per_device_eval_batch_size=8 \
    --num_train_epochs 7 \
    --predict_with_generate --save_strategy no \
    --metric_for_best_model bleu --overwrite_output_dir \
    --logging_dir ./mBART50_san_hin/log --report_to tensorboard
```

### 4. Inference
```bash
python3 transformers/examples/pytorch/translation/run_translation.py \
    --model_name_or_path checkpoint/mBART50_san_hin/ \
    --do_predict \
    --source_lang ma_XX --target_lang hi_IN \
    --validation_file /home/SanHinCorp1.0/mBART50/data/dev/dev.json \
    --test_file /home/SanHinCorp1.0/mBART50/data/test/test.json \
    --output_dir /home/SanHinCorp1.0/mBART50/output \
    --per_device_eval_batch_size=4 \
    --predict_with_generate --overwrite_output_dir
```

### 5. Compute BLEU4, chrF2, TER, COMET, METEOR, BERTScore Scores
```bash
cd /home/SanHinCorp1.0/mBART50/output
python BLEU_chrF_TER.py
python COMET_METEOR.py
python BERTscore.py
```


## mT5 Model

### 1. Create Conda Environment
```bash
mkdir SanHinCorp1.0
cd SanHinCorp1.0
conda env create -f  mT5_environment.yml
conda activate mT5
mkdir mT5
cd mT5
```

### 2. Install Requirements
```bash
pip install -r requirement.txt
```

### 3. Training
```bash
python3 transformers/examples/pytorch/translation/run_translation.py \
    --model_name_or_path google/mt5-base \
    --do_train --do_eval \
    --source_lang ma --target_lang hi --source_prefix "<2hi> " \
    --train_file /home/SanHinCorp1.0/mT5/data/train/train_san_hin.json \
    --validation_file /home/SanHinCorp1.0/mT5/data/test/test_san_hin.json \
    --test_file /home/SanHinCorp1.0/mT5/data/test/test_san_hin.json \
    --output_dir checkpoints/mT5_mahi/ \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=4 \
    --num_train_epochs 7 \
    --predict_with_generate --save_strategy no \
    --metric_for_best_model bleu --overwrite_output_dir
```

### 4. Inference & BLEU Score Calculation
```bash
python3 transformers/examples/pytorch/translation/run_translation.py \
    --model_name_or_path checkpoints/mT5_mahi \
    --do_predict \
    --source_lang ma --target_lang hi --source_prefix "<2hi> " \
    --validation_file /home/SanHinCorp1.0/mT5/data/test/test_san_hin.json \
    --test_file /home/SanHinCorp1.0/mT5/data/test/test_san_hin.json \
    --output_dir checkpoints/mT5_mahi/output \
    --per_device_eval_batch_size=4 \
    --predict_with_generate --overwrite_output_dir
```

### 5. ### 5. Compute BLEU4, chrF2, TER, COMET, METEOR, BERTScore Scores
```bash
cd /home/SanHinCorp1.0/mT5/output
python BLEU_chrF_TER.py
python COMET_METEOR.py
python BERTscore.py
```

---


## NLLB-200 Model

### 1. Create Environment and Project Structure
```bash
mkdir SanHinCorp1.0
cd SanHinCorp1.0
conda env create -f  NLLB-200_environment.yml
conda activate NLLB-200
mkdir NLLB-200
cd NLLB-200
```

### 2. Download nllb-200-distilled-600M Model from HuggingFace, Ensure that all dataset files are in huggingface dataset format (e.g., `train`, `valid`, `test`).

```bash
cd /home/SanHin1.0/NLLB-200/scripts
python download.py
python convert_dataset.py
```
### 3. Training
```bash
python train.py
```
### 4. Inference

```bash
python inference.py
```
### 5. Compute BLEU4, chrF2, TER, COMET, METEOR, BERTScore Scores
```bash
python BLEU_chrF_TER.py
python COMET_METEOR.py
python BERTscore.py
```

## OPUS-MT model

### 1. Create Conda Environment
```bash
mkdir SanHinCorp1.0
cd SanHinCorp1.0
conda env create -f  OPUS-MT_environment.yml
conda activate opus_env
mkdir OPUS-MT
cd OPUS-MT

```

### 2. Training

```bash
mkdir scripts
cd scripts 
python finetune.py

```

### 3. Inference
```bash
python inference.py
```
### 4. Evaluation: Compute BLEU4, chrF2, TER, COMET, METEOR, BERTScore Scores

```bash
cd test
python BLEU_chrF_TER.py
python COMET_METEOR.py
python BERTscore.py
```


