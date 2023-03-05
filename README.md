# CSCI 6908 - Special Graduate Topics in Computer Science 2023: RobustQA track

### Low-Rank Adaptation of Large Language Models for Robust Question Answering
- In this repo, we applied Low-Rank Adaptation technique in fine-tuning stage (stage 2) on balanced set of domains (`indomain_train` and `oodomain_train`).
- We have 3 domains in `indomain_train` and 3 target domains in `oodomain_train`. By this splitting, we can achieve better performance on both in-domains and out-of-domains.
- Inspiring by [Balanced Set Finetuning](https://arxiv.org/abs/2003.06957), in the second stage after normal pretraining DistilBERT on `indomain_train`, we fine-tuned the trained model on 127 samples domain. 
- Low-Rank Adaptation technique simply plug two new matrices weight per attention module in DistilBERT and only train them during fine-tuning allow efficient domain adaptation. For more details, please see the paper: https://doi.org/10.48550/arXiv.2106.09685.

## How it's implemented
We used a package called [PEFT] for efficient fine-tuning. To learn more about [PEFT], please visit this git repo: https://github.com/huggingface/peft
```
if args.do_train_stage_2:
        model_config = model.config.to_dict()
        peft_config = LoraConfig(
            task_type="Question-Answering", inference_mode=False, r=24, lora_alpha=32, lora_dropout=0.2
        )
        peft_config.target_modules = ['q_lin', 'v_lin'] ## Apply LoRA to Query and Value Matrices in DistilBERT
        peft_config = _prepare_lora_config(peft_config, model_config)
        model = PeftModeForQuestionAnswering(model, peft_config) ## Load DistilBERT to customized QAPeft
        print(f'Using LORA model: {model.__class__.__name__}')
        print(f'Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
        print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')
```

## Text Augmentation for Robust Question Answering
We applied different text augmentation strategies to the question to make the model more robust. All strategies were implemented in `augment/` and `data.py` to apply the augmentation to the dataset. The following perturbation methods are available to augment SQuAD-like data:
- Synonym Replacement (SR) via 1) [constrained word2vec](https://arxiv.org/pdf/1603.00892.pdf), and 2) MLM using BERT
```diff
- (original)  How many species of plants were [recorded] in Egypt?
+ (augmented) How many species of plants were [registered] in Egypt?
```
- Random Deletion (RD) using entity-aware term selection
```diff
- (original)  How many species of plants [were] recorded in Egypt?
+ (augmented) How many species of plants [] recorded in Egypt?
```
- Random Repetition (RR) using entity-aware term selection
```diff
- (original)  How many species of plants [were] recorded in Egypt?
+ (augmented) How many species of plants [were were] recorded in Egypt?
```
- Random Misspelling (RM) using open-source common misspellings datasets
    -- *sources: [wiki](https://en.wikipedia.org/wiki/Wikipedia:Lists_of_common_misspellings), [brikbeck](https://www.dcs.bbk.ac.uk/~ROGER/corpora.html)*
```diff
- (original)  How [many] species of plants were recorded in Egypt?
+ (augmented) How [mony] species of plants were recorded in Egypt?
```
## Multi-domains Evaluation
We have changed the codebase in `train.py` so that it can evaluate multiple domains during the training for better understanding the behaviour of a model.

## Starter code for robustqa track
- Download datasets from [here](https://drive.google.com/file/d/1Fv2d30hY-2niU7t61ktnMsi_HUXS6-Qx/view?usp=sharing)
- Download support from [here](https://github.com/searchableai/KitanaQA/tree/master/src/kitanaqa/support)
- Setup environment with `conda env create -f environment.yml`
- ``pip install loralib``
- ``pip install Peft``
- Train the base system with `python train.py --do-train --eval-every 2000 --run-name baseline`
- Train stage 2 with Low-Rank Adaptation finetuning by executing `python train.py --do-train --eval-every 20 --run-name stage2 --pretrained-model-name-or-path save/baseline-01/checkpoint/ --do-train-stage-2 --train-datasets squad,nat_questions,newsqa,race,relation_extraction,duorc --num-epochs 4 --lr 3e-5`
- Evaluate the system on test set with `python train.py --do-eval --sub-file mtl_submission.csv --save-dir save/baseline-01`
- Upload the csv file in `save/baseline-01` to the test leaderboard. For the validation leaderboard, run `python train.py --do-eval --sub-file mtl_submission_val.csv --save-dir save/baseline-01 --eval-dir datasets/oodomain_val`

## Code Structure
.
├── augment/
├── datasets/
│ ├── indomain_train/
│ ├── indomain_val/
│ ├── oodomain_train/
│ ├── oodomain_test/
│ └── oodomain_val/
├── support/
├── convert_to_squad.py
├── data.py
├── environment.yml
├── lora.py
├── train.py
└── util.py
└── util.py


## Results
| Method | Dataset | F1 | EM |
| --- | --- | --- | --- |
| Baseline Method | In-domains Dataset | 70.12 | 51.00 |
| Baseline Method | Out-of-domains Dataset | 44.73 | 29.58 |
| LoRA | In-domains Dataset | 70.24 | 52.40 |
| LoRA | Out-of-domains Dataset | 49.56 | 33.00 |
| LoRA + Aug | In-domains Dataset | 70.54 | 54.40 |
| LoRA + Aug | Out-of-domains Dataset | 50.86 | 34.40 |

