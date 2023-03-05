import argparse
import json
import os
from collections import OrderedDict
import torch
import csv
import util
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering
from transformers import AdamW
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from args import get_train_test_args
import random
from tqdm import tqdm
from torchsummary import summary
from data import read_and_process
from peft.mapping import _prepare_lora_config
from peft import LoraConfig
from lora import PeftModeForQuestionAnswering

class Trainer():
    def __init__(self, args, log):
        self.lr = args.lr
        self.num_epochs = args.num_epochs
        self.device = args.device
        self.eval_every = args.eval_every
        self.path = os.path.join(args.save_dir, 'checkpoint')
        self.num_visuals = args.num_visuals
        self.save_dir = args.save_dir
        self.log = log
        self.visualize_predictions = args.visualize_predictions
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def save(self, model):
        model.save_pretrained(self.path)

    def evaluate(self, model, data_loader, data_dict, return_preds=False, split='validation'):
        device = self.device

        model.eval()
        pred_dict = {}
        all_start_logits = []
        all_end_logits = []
        with torch.no_grad(), \
                tqdm(total=len(data_loader.dataset)) as progress_bar:
            for batch in data_loader:
                # Setup for forward
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                batch_size = len(input_ids)
                outputs = model(input_ids, attention_mask=attention_mask)
                # Forward
                start_logits, end_logits = outputs.start_logits, outputs.end_logits
                # TODO: compute loss

                all_start_logits.append(start_logits)
                all_end_logits.append(end_logits)
                progress_bar.update(batch_size)

        # Get F1 and EM scores
        start_logits = torch.cat(all_start_logits).cpu().numpy()
        end_logits = torch.cat(all_end_logits).cpu().numpy()
        preds = util.postprocess_qa_predictions(data_dict,
                                                 data_loader.dataset.encodings,
                                                 (start_logits, end_logits))
        if split == 'validation':
            results = util.eval_dicts(data_dict, preds)
            results_list = [('F1', results['F1']),
                            ('EM', results['EM'])]
        else:
            results_list = [('F1', -1.0),
                            ('EM', -1.0)]
        results = OrderedDict(results_list)
        if return_preds:
            return preds, results
        return results

    def train(self, model, train_dataloader, eval_dataloader, base_eval_dataloader, val_dict, base_val_dict):
        device = self.device
        model.to(device)
        optim = AdamW(model.parameters(), lr=self.lr)
        global_idx = 0
        best_scores = {'F1': -1.0, 'EM': -1.0}
        tbx = SummaryWriter(self.save_dir)

        for epoch_num in range(self.num_epochs):
            self.log.info(f'Epoch: {epoch_num}')
            with torch.enable_grad(), tqdm(total=len(train_dataloader.dataset)) as progress_bar:
                for batch in train_dataloader:
                    optim.zero_grad()
                    model.train()
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    start_positions = batch['start_positions'].to(device)
                    end_positions = batch['end_positions'].to(device)
                    outputs = model(input_ids, attention_mask=attention_mask,
                                    start_positions=start_positions,
                                    end_positions=end_positions)
                    loss = outputs[0]
                    loss.backward()
                    optim.step()
                    progress_bar.update(len(input_ids))
                    progress_bar.set_postfix(epoch=epoch_num, NLL=loss.item())
                    tbx.add_scalar('train/NLL', loss.item(), global_idx)
                    if (global_idx % self.eval_every) == 0:
                        self.log.info(f'Evaluating at step {global_idx}...')
                        preds, curr_score = self.evaluate(model, eval_dataloader, val_dict, return_preds=True)
                        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in curr_score.items())
                        self.log.info('Visualizing in TensorBoard...')
                        for k, v in curr_score.items():
                            tbx.add_scalar(f'val/{k}', v, global_idx)
                        self.log.info(f'Eval {results_str}')
                        if self.visualize_predictions:
                            util.visualize(tbx,
                                           pred_dict=preds,
                                           gold_dict=val_dict,
                                           step=global_idx,
                                           split='val',
                                           num_visuals=self.num_visuals)
                        if curr_score['F1'] >= best_scores['F1']:
                            best_scores = curr_score
                            self.save(model)
                        
                        preds, curr_score = self.evaluate(model, base_eval_dataloader, base_val_dict, return_preds=True)
                        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in curr_score.items())
                        self.log.info(f'Base Eval {results_str}')
                        
                    global_idx += 1
        return best_scores

def get_dataset(args, datasets, data_dir, tokenizer, split_name, stage_2=False, stage_2_samples=127):
    hparams = {
            "num_replacements":2,
            "sample_ratio":0.0,
            "p_replace":0.15,
            "p_dropword":0.1,
            "p_misspelling":0.1,
            "sampling_strategy":'topK',
            "sampling_k":2
        }
    datasets = datasets.split(',')
    dataset_dict = None
    dataset_name=''
    for dataset in datasets:
        dataset_name += f'_{dataset}'
        if dataset in ["squad", "nat_questions", "newsqa"]:
            dataset_dict_curr = util.read_squad(f'{data_dir}/{dataset}')
        else:
            data_dir = "datasets/oodomain_train"
            dataset_dict_curr = util.read_squad(f'{data_dir}/{dataset}')
        ## Balanced merging finetuning
        if stage_2:
            hparams = {
                "num_replacements":2,
                "sample_ratio":0.0,
                "p_replace":0.1,
                "p_dropword":0.1,
                "p_misspelling":0.1,
                "sampling_strategy":'topK',
                "sampling_k":2
            }
            stage_2_samples_id = random.sample(range(len(dataset_dict_curr['question'])), stage_2_samples)
            for key in dataset_dict_curr.keys():
                dataset_dict_curr[key] = [dataset_dict_curr[key][i] for i in stage_2_samples_id]
        dataset_dict = util.merge(dataset_dict, dataset_dict_curr)
    data_encodings = read_and_process(args, tokenizer, dataset_dict, data_dir, dataset_name, split_name, hparams)
    return util.QADataset(data_encodings, train=(split_name=='train')), dataset_dict

def main():
    # define parser and arguments
    args = get_train_test_args()

    util.set_seed(args.seed)
    model = DistilBertForQuestionAnswering.from_pretrained(args.pretrained_model_name_or_path)
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    if args.do_train_stage_2:
        model_config = model.config.to_dict()
        peft_config = LoraConfig(
            task_type="Question-Answering", inference_mode=False, r=24, lora_alpha=32, lora_dropout=0.2
        )
        peft_config.target_modules = ['q_lin', 'v_lin']
        peft_config = _prepare_lora_config(peft_config, model_config)
        model = PeftModeForQuestionAnswering(model, peft_config)
        print(f'Using LORA model: {model.__class__.__name__}')
        print(f'Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
        print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')
        
    if args.do_train:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        args.save_dir = util.get_save_dir(args.save_dir, args.run_name)
        log = util.get_logger(args.save_dir, 'log_train')
        log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
        log.info("Preparing Training Data...")
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        trainer = Trainer(args, log)
        if not args.do_train_stage_2:
            train_dataset, _ = get_dataset(args, args.train_datasets, args.train_dir, tokenizer, 'train')
        else:
            train_dataset, _ = get_dataset(args, args.train_datasets, args.train_dir, tokenizer, 'train', stage_2=True, stage_2_samples=args.stage_2_samples)
        log.info("Preparing Validation Data...")
        val_dataset, val_dict = get_dataset(args, args.eval_datasets, args.val_dir, tokenizer, 'val')
        base_val_dataset, base_val_dict = get_dataset(args, 'squad,nat_questions,newsqa', 'datasets/indomain_val', tokenizer, 'val')
        train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                sampler=RandomSampler(train_dataset))
        val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                sampler=SequentialSampler(val_dataset))
        base_val_loader = DataLoader(base_val_dataset,
                                batch_size=args.batch_size,
                                sampler=SequentialSampler(base_val_dataset))
        
        best_scores = trainer.train(model, train_loader, val_loader, base_val_loader, val_dict, base_val_dict)
    if args.do_eval:
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        split_name = 'test' if 'test' in args.eval_dir else 'validation'
        log = util.get_logger(args.save_dir, f'log_{split_name}')
        trainer = Trainer(args, log)
        checkpoint_path = os.path.join(args.save_dir, 'checkpoint')
        model = DistilBertForQuestionAnswering.from_pretrained(checkpoint_path)
        model.to(args.device)
        eval_dataset, eval_dict = get_dataset(args, args.eval_datasets, args.eval_dir, tokenizer, split_name)
        eval_loader = DataLoader(eval_dataset,
                                 batch_size=args.batch_size,
                                 sampler=SequentialSampler(eval_dataset))
        eval_preds, eval_scores = trainer.evaluate(model, eval_loader,
                                                   eval_dict, return_preds=True,
                                                   split=split_name)
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in eval_scores.items())
        log.info(f'Eval {results_str}')
        # Write submission file
        sub_path = os.path.join(args.save_dir, split_name + '_' + args.sub_file)
        log.info(f'Writing submission file to {sub_path}...')
        with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
            csv_writer = csv.writer(csv_fh, delimiter=',')
            csv_writer.writerow(['Id', 'Predicted'])
            for uuid in sorted(eval_preds):
                csv_writer.writerow([uuid, eval_preds[uuid]])


if __name__ == '__main__':
    main()
