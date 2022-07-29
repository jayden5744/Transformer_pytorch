import math
import os
import wandb
from typing import Tuple
from omegaconf import DictConfig


import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import CrossEntropyLoss

from torch.utils.data import DataLoader

from src.model import Encoder, Decoder, Transformer
from src.utils.data_helper import TransformerDataset, create_or_get_voca
from src.utils.metrics import calculate_bleu
from src.utils.utils import count_parameters, EarlyStopping
from src.utils.weight_initialization import select_weight_initialize_method


class Trainer(object):
    def __init__(self, cfg: DictConfig):
        self.args = cfg
        self.device = self.get_device()
        self.optimizer = None
        self.src_voca, self.trg_voca = self.get_voca()
        self.train_loader, self.valid_loader = self.get_loader()
        self.criterion = CrossEntropyLoss(
            ignore_index=self.args.data.pad_id,
            label_smoothing=self.args.trainer.label_smoothing_value
        )
        self.early_stopping = EarlyStopping(patience=self.args.trainer.early_stopping, verbose=True)
        wandb.init()
        wandb.config.update(self.args)

    def train(self):
        model = self.get_model()
        print(f'The model has {count_parameters(model):,} trainable parameters')

        select_weight_initialize_method(
            method=self.args.trainer.weight_init,
            distribution=self.args.trainer.weight_distribution,
            model=model
        )

        self.optimizer = self.init_optimizer(model)
        wandb.watch(model)

        epoch_step = len(self.train_loader) + 1
        total_step = self.args.trainer.epochs * epoch_step
        step = 0
        for epoch in range(self.args.trainer.epoch):
            for i, data in enumerate(self.train_loader, 1):
                try:
                    self.optimizer.zero_grad()
                    src_input, tar_input, tar_output = data
                    output, _ = model(src_input, tar_input)
                    loss, accuracy, ppl = self.calculate_loss(output, tar_output)

                    if step % self.args.trainer.train_step_print == 0:
                        wandb.log({
                            ""
                        })
                        print('[Train] epoch : {0:2d}  iter: {1:4d}/{2:4d}  step : {3:6d}/{4:6d}  '
                              '=>  loss : {5:10f}  accuracy : {6:12f}  PPL : {7:6f}'
                              .format(epoch, i, epoch_step, step, total_step, loss.item(), accuracy.item(), ppl))

                    if step % self.args.trainer.valid_step_print == 0:
                        with torch.no_grad():
                            val_loss, val_accuracy, val_ppl = self.valid(model)
                            print('[Val] epoch : {0:2d}  iter: {1:4d}/{2:4d}  step : {3:6d}/{4:6d}  '
                                  '=>  loss : {5:10f}  accuracy : {6:12f}   PPL : {7:10f}'
                                  .format(epoch, i, epoch_step, step, total_step, val_loss, val_accuracy, val_ppl))
                            # todo: early stopping 수정
                            self.early_stopping(val_loss, model, step)

                    if step % self.args.trainer.save_step == 0:
                        self.save_model(model, epoch, step)

                    if self.early_stopping.early_stop:
                        break

                    loss.backward()
                    self.optimizer.step()
                    step += 1

                except Exception as e:
                    self.save_model(model, epoch, step)
                    raise e

            if self.early_stopping.early_stop:
                break

    def get_encoder_params(self):
        return {
            'input_dim': self.args.model.enc_vocab_size,
            'hid_dim': self.args.model.enc_hidden_dim,
            'n_layers': self.args.model.enc_layers,
            'n_heads': self.args.model.enc_heads,
            'head_dim': self.args.model.enc_head_dim,
            'pf_dim': self.args.model.enc_ff_dim,
            'dropout': self.args.model.dropout_rate,
            'max_length': self.args.model.max_sequence_len,
            'padding_id': self.args.data.pad_id,
        }

    def get_decoder_params(self):
        return {
            'input_dim': self.args.model.dec_vocab_size,
            'hid_dim': self.args.model.dec_hidden_dim,
            'n_layers': self.args.model.dec_layers,
            'n_heads': self.args.model.dec_heads,
            'head_dim': self.args.model.dec_head_dim,
            'pf_dim': self.args.model.dec_ff_dim,
            'dropout': self.args.model.dropout_rate,
            'max_length': self.args.model.max_sequence_len,
            'padding_id': self.args.data.pad_id,
        }

    def get_model(self) -> nn.Module:
        encoder = Encoder(**self.get_encoder_params())
        decoder = Decoder(**self.get_decoder_params())
        model = Transformer(encoder, decoder)
        if torch.cuda.is_available() and self.args.trainer.is_gpu:
            model = nn.DataParallel(model)
            model.cuda()
        model.train()
        return model

    def get_device(self):
        return torch.device('cuda' if torch.cuda.is_available() and self.args.trainer.is_gpu else 'cpu')

    def get_voca(self):
        return create_or_get_voca(
            save_path=self.args.data.dictionary_path,
            src_corpus_path=self.args.data.src_train_path,
            trg_corpus_path=self.args.data.trg_train_path,
            src_vocab_size=self.args.model.enc_vocab_size,
            trg_vocab_size=self.args.model.dec_vocab_size,
            bos_id=self.args.data.bos_id,
            eos_id=self.args.data.eos_id,
            unk_id=self.args.data.unk_id,
            pad_id=self.args.data.pad_id
        )

    def get_loader(self) -> Tuple[DataLoader, DataLoader]:
        train_dataset = TransformerDataset(
            x_path=self.args.data.src_train_path,
            y_path=self.args.data.trg_train_path,
            src_voc=self.src_voca,
            trg_voc=self.trg_voca,
            sequence_size=self.args.model.max_sequence_len,
        )
        valid_dataset = TransformerDataset(
            x_path=self.args.data.src_val_path,
            y_path=self.args.data.trg_val_path,
            src_voc=self.src_voca,
            trg_voc=self.trg_voca,
            sequence_size=self.args.model.max_sequence_len,
        )

        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        valid_sampler = torch.utils.data.RandomSampler(valid_dataset)

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.args.trainer.batch_size,
            sampler=train_sampler
        )

        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=self.args.trainer.batch_size,
            sampler=valid_sampler
        )
        return train_loader, valid_loader

    def init_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        if self.args.trainer.optimizer == "Adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=self.args.trainer.learning_rate,
                betas=(self.args.trainer.optimizer_b1, self.args.trainer.optimizer_b2),
                eps=self.args.trainer.optimizer_e
            )

        else:
            raise ValueError("trainer param `optimizer` is one of [Adam] ")

    def calculate_loss(self, pred, tar) -> Tuple[Tensor, Tensor, float]:
        pred = pred.view(-1, pred.size(-1))
        tar = tar.view(-1).to(self.device)

        loss = self.criterion(pred, tar)
        ppl = math.exp(loss.item())

        indices = pred.max(-1)[1]
        invalid_targets = tar.eq(self.args.data.pad_id)
        equal = indices.eq(tar)
        total = 0
        for i in invalid_targets:
            if i == 0:
                total += 1

        accuracy = torch.div(equal.masked_fill_(invalid_targets, 0).long().sum().to(dtype=torch.float32), total)
        return loss, accuracy, ppl

    def valid(self, model: nn.Module) -> Tuple[float, float, float]:
        model.eval()
        total_loss, total_accuracy, total_ppl = 0, 0, 0

        for data in self.valid_loader:
            src_input, tar_input, tar_output = data
            output, _ = model(src_input, tar_input)
            loss, accuracy, ppl = self.calculate_loss(output, tar_output)
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            total_ppl += ppl

        prob = output.squeeze(0).max(dim=-1, keepdim=False)[1]
        indices = prob.data.tolist()[0]
        a = src_input[0].tolist()
        b = tar_output[0].tolist()

        output_sentence = self.tensor2sentence(indices, self.trg_voca)
        target_sentence = self.tensor2sentence(b, self.trg_voca)
        bleu_score = calculate_bleu(output_sentence, target_sentence)

        print("-------test-------")
        print(f"Source      : {self.tensor2sentence(a, self.src_voca)}")
        print(f"Predicted   : {output_sentence}")
        print(f"Target      : {target_sentence}")
        print(f"BLEU Score  : {bleu_score}")

        model.train()
        avg_loss = total_loss / len(self.valid_loader)
        avg_accuracy = total_accuracy / len(self.valid_loader)
        avg_ppl = total_ppl / len(self.valid_loader)
        return avg_loss, avg_accuracy, avg_ppl

    def tensor2sentence(self, indices: torch.Tensor, vocabulary) -> str:
        translation_sentence = []
        for idx in indices:
            word = vocabulary.IdToPiece(idx)
            if word == self.args.data.eos_id or word == self.args.data.pad_id:
                break
            translation_sentence.append(word)
        return ''.join(translation_sentence).replace('▁', ' ').strip()

    def save_model(self, model: nn.Module, epoch: int, step: int) -> None:
        model_name = '{0:06d}_transformer.pth'.format(step)
        model_path = os.path.join(self.args.data.model_path, model_name)
        torch.save(
            {
                'epoch': epoch,
                'step': step,
                'encoder_parameter': self.get_encoder_params(),
                'decoder_parameter': self.get_decoder_params(),
                'model_state_dict': model.state_dict()
            }, model_path
        )
