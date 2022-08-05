import math
import os
from abc import abstractmethod, ABC

import wandb
from typing import Tuple, Dict, List
from omegaconf import DictConfig

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import CrossEntropyLoss

from torch.utils.data import DataLoader

from src.model import Encoder, Decoder, Transformer
from src.utils.data_helper import TransformerDataset, create_or_get_voca, TransformerTestDataset
from src.utils.metrics import calculate_bleu
from src.utils.utils import count_parameters, EarlyStopping
from src.utils.weight_initialization import select_weight_initialize_method


class AbstractTranslation:
    def __init__(self):
        self.args = None

    @abstractmethod
    def get_model(self):
        pass

    def get_device(self):
        return torch.device('cuda' if torch.cuda.is_available() and self.args.trainer.is_gpu else 'cpu')

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

    def tensor2sentence(self, indices: torch.Tensor, vocabulary) -> str:
        translation_sentence = []
        for idx in indices:
            word = vocabulary.IdToPiece(idx)
            if word == self.args.data.eos_id or word == self.args.data.pad_id:
                break
            translation_sentence.append(word)
        return ''.join(translation_sentence).replace('▁', ' ').strip()


class Trainer(AbstractTranslation, ABC):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.args = cfg
        self.optimizer = None
        self.device = self.get_device()
        self.src_voca, self.trg_voca = self.get_voca()

        self.train_loader, self.valid_loader = self.get_loader()
        self.criterion = CrossEntropyLoss(
            ignore_index=self.args.data.pad_id,
            label_smoothing=self.args.trainer.label_smoothing_value
        )
        self.early_stopping = EarlyStopping(patience=self.args.trainer.early_stopping, args=cfg, verbose=True)
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
                            "Train loss": loss.item(),
                            "Train Accuracy": accuracy.item(),
                            "Train PPL": ppl
                        })
                        print('[Train] epoch : {0:2d}  iter: {1:4d}/{2:4d}  step : {3:6d}/{4:6d}  '
                              '=>  loss : {5:10f}  accuracy : {6:12f}  PPL : {7:6f}'
                              .format(epoch, i, epoch_step, step, total_step, loss.item(), accuracy.item(), ppl))

                    if step % self.args.trainer.valid_step_print == 0:
                        with torch.no_grad():
                            val_loss, val_accuracy, val_ppl = self.valid(model)
                            wandb.log({
                                "Valid loss": val_loss,
                                "Valid Accuracy": val_accuracy,
                                "Valid PPL": val_ppl
                            })
                            print('[Val] epoch : {0:2d}  iter: {1:4d}/{2:4d}  step : {3:6d}/{4:6d}  '
                                  '=>  loss : {5:10f}  accuracy : {6:12f}   PPL : {7:10f}'
                                  .format(epoch, i, epoch_step, step, total_step, val_loss, val_accuracy, val_ppl))

                            self.early_stopping(val_loss, model, epoch, step)

                    if step % self.args.trainer.save_step == 0:
                        self.save_model(model, '{0:06d}_transformer.pth'.format(step), epoch)

                    if self.early_stopping.early_stop:
                        break

                    loss.backward()
                    self.optimizer.step()
                    step += 1

                except Exception as e:
                    self.save_model(model, '{0:06d}_transformer.pth'.format(step), epoch)
                    raise e

            if self.early_stopping.early_stop:
                break

    def get_model(self) -> nn.Module:
        encoder = Encoder(**self.get_encoder_params())
        decoder = Decoder(**self.get_decoder_params())
        model = Transformer(encoder, decoder)
        if torch.cuda.is_available() and self.args.trainer.is_gpu:
            model = nn.DataParallel(model)
            model.cuda()
        model.train()
        return model

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

        avg_loss = total_loss / len(self.valid_loader)
        avg_accuracy = total_accuracy / len(self.valid_loader)
        avg_ppl = total_ppl / len(self.valid_loader)
        model.train()
        return avg_loss, avg_accuracy, avg_ppl

    def save_model(self, model: nn.Module, model_name: str, epoch: int) -> None:
        model_path = os.path.join(self.args.data.model_path, model_name)
        torch.save(
            {
                'epoch': epoch,
                "data": self.args["data"],
                "trainer": self.args["trainer"],
                "model": self.args["model"],
                'model_state_dict': model.state_dict()
            }, model_path
        )


class Inference(AbstractTranslation, ABC):
    def __init__(self, check_point: str, is_gpu: bool = False):
        super().__init__()
        self.check_point = torch.load(check_point)
        self.args = self.get_args()
        self.args.trainer.is_gpu = is_gpu
        self.model = self.get_model()
        self.src_voca, self.trg_voca = self.get_voca()
        self.device = self.get_device()

    def get_args(self) -> DictConfig:
        return DictConfig({"data": self.check_point["data"], "model": self.check_point["model"],
                           "trainer": self.check_point["trainer"]})

    def get_model(self) -> nn.Module:
        encoder = Encoder(**self.get_encoder_params())
        decoder = Decoder(**self.get_decoder_params())
        model = Transformer(encoder, decoder)
        if torch.cuda.is_available() and self.args.trainer.is_gpu:
            model = nn.DataParallel(model)
            model.cuda()

        model.load_state_dict(self.check_point["model_state_dict"])
        model.eval()
        return model

    def inference(self, sentence: str, debug: bool = True) -> str:
        enc_input = self.encoder_input_to_vector(sentence)
        decoder_output = self.greedy_decoder(enc_input)
        output_sentence = self.tensor2sentence(decoder_output, self.trg_voca)
        if debug:
            print(f"Source : {sentence}")
            print(f"Target : {output_sentence}")
        return output_sentence

    def batch_inference(self, src_file_path: str, batch_size: int, debug: bool = True) -> List[str]:
        dataset = TransformerTestDataset(x_path=src_file_path, src_voc=self.src_voca,
                                         sequence_size=self.args.model.max_sequence_len)
        loader = DataLoader(dataset, batch_size=batch_size)
        predicts = []
        try:
            for i, data in enumerate(loader):
                src_input = data
                decoder_output = self.greedy_decoder(src_input)
                for src_indices, out_indices in zip(src_input.tolist(), decoder_output):
                    out_sentence = self.tensor2sentence(out_indices, self.trg_voca)
                    if debug:
                        src_sentence = self.tensor2sentence(src_indices, self.src_voca)
                        print("------ Inference ------")
                        print(f"Source : {src_sentence}")
                        print(f"Predict : {out_sentence}")
                    predicts.append(out_sentence)

        except KeyboardInterrupt:
            pass
        return predicts

    def evaluate(self, src_file_path: str, trg_file_path: str, batch_size: int, debug: bool = True):
        pass

    def greedy_decoder(self, enc_input: Tensor) -> Tensor:
        batch_size = enc_input.size(0)
        enc_outputs, _ = self.model.module.encoder(enc_input)
        dec_input = torch.LongTensor(batch_size, self.args.model.max_sequence_len).fill_(4).to(self.device)
        decoded_batch = torch.zeros((batch_size, self.args.model.max_sequence_len))

        next_symbol = self.args.data.bos_id

        for i in range(0, self.args.model.max_sequence_len):
            dec_input[:, i] = next_symbol
            dec_outputs, _ = self.model.module.decoder(dec_input, enc_input, enc_outputs)
            prob = dec_outputs.squeeze(0).max(dim=-1, keepdim=False)[1]
            if batch_size == 1:
                prob = prob.unsqueeze(dim=0)

            next_symbol = torch.LongTensor(prob.data[:, i].tolist()).to(self.device)

            if batch_size == 1:
                next_symbol = next_symbol.unsqueeze(dim=0)

            decoded_batch[:, i] = next_symbol
        return decoded_batch

    def encoder_input_to_vector(self, sentence: str) -> torch.Tensor:
        idx_list = self.src_voca.EncodeAsIds(sentence)
        idx_list = self.padding(idx_list, self.src_voca['<pad>'])
        return torch.tensor([idx_list]).to(self.device)

    def padding(self, idx_list: List[int], padding_id: int) -> List[int]:
        length = len(idx_list)
        if length < self.args.model.max_sequence_len:
            idx_list = idx_list + [padding_id for _ in range(self.args.model.max_sequence_len - len(idx_list))]
        else:
            idx_list = idx_list[:self.args.model.max_sequence_len]
        return idx_list
