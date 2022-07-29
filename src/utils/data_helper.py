import os
import shutil
from abc import *
from typing import Tuple, Any, List
import sentencepiece as spm

import torch
from torch import Tensor
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_or_get_voca(
        save_path: str,
        src_corpus_path: str = None,
        trg_corpus_path: str = None,
        src_vocab_size: int = 4000,
        trg_vocab_size: int = 4000,
        bos_id: int = 0,
        eos_id: int = 1,
        unk_id: int = 2,
        pad_id: int = 3
) -> Tuple[Any, Any]:
    """
    create or get vocabulary function
    Args:
        save_path(str): Dictionary 파일을 저장할 folder path
        src_corpus_path(str): source corpus path
        trg_corpus_path(str): target corpus path
        src_vocab_size(int): source vocab size (default: 4000)
        trg_vocab_size(int): target vocab size (default: 4000)
        bos_id(int): begin of sentence token id (default: 0)
        eos_id(int): end of sentence token id (default: 1)
        unk_id(int): unknown token id (default: 2)
        pad_id(int): padding token id (default: 3)

    Returns:
        Tuple[Any, Any]: source processor, target processor

    """
    src_corpus_prefix = f'src_corpus_{src_vocab_size}'
    trg_corpus_prefix = f'trg_corpus_{trg_vocab_size}'

    if src_corpus_path and trg_corpus_path:
        templates = '--input={} --model_prefix={} --vocab_size={} ' \
                    '--bos_id={} --eos_id={} --unk_id={} --pad_id={}'

        src_model_train_cmd = templates.format(src_corpus_path, src_corpus_prefix, src_vocab_size,
                                               bos_id, eos_id, unk_id, pad_id)
        trg_model_train_cmd = templates.format(trg_corpus_path, trg_corpus_prefix, trg_vocab_size,
                                               bos_id, eos_id, unk_id, pad_id)

        spm.SentencePieceTrainer.Train(src_model_train_cmd)
        spm.SentencePieceTrainer.Train(trg_model_train_cmd)

        shutil.move(src_corpus_prefix + '.model', save_path)
        shutil.move(src_corpus_prefix + '.vocab', save_path)
        shutil.move(trg_corpus_prefix + '.model', save_path)
        shutil.move(trg_corpus_prefix + '.vocab', save_path)

    src_sp = spm.SentencePieceProcessor()
    trg_sp = spm.SentencePieceProcessor()
    src_sp.load(os.path.join(save_path, src_corpus_prefix + '.model'))
    trg_sp.load(os.path.join(save_path, trg_corpus_prefix + '.model'))
    return src_sp, trg_sp


class AbstractDataset(Dataset, metaclass=ABCMeta):
    def __init__(self, x_path: str, y_path: str, src_voc: Any, trg_voc: Any, sequence_size: int):
        self.x = open(x_path, 'r', encoding='utf-8').readlines()  # korean data 위치
        self.y = open(y_path, 'r', encoding='utf-8').readlines()  # English data 위치
        self.src_voc = src_voc
        self.trg_voc = trg_voc
        self.sequence_size = sequence_size

        self.BOS, self.EOS, self.PAD = None, None, None

    def __len__(self):
        if len(self.x) != len(self.y):
            raise IndexError('not equal x_path, y_path line size')
        return len(self.x)

    def encoder_input_to_vector(self, sentence: str) -> torch.Tensor:
        idx_list = self.src_voc.EncodeAsIds(sentence)  # str -> idx
        idx_list = self.padding(idx_list, self.PAD)  # padding 삽입
        return torch.tensor(idx_list).to(device)

    def decoder_input_to_vector(self, sentence: str) -> torch.Tensor:
        idx_list = self.trg_voc.EncodeAsIds(sentence)  # str -> idx
        idx_list.insert(0, self.BOS)  # Start Token 삽입
        idx_list = self.padding(idx_list, self.PAD)  # padding 삽입
        return torch.tensor(idx_list).to(device)

    def decoder_output_to_vector(self, sentence: str) -> torch.Tensor:
        idx_list = self.trg_voc.EncodeAsIds(sentence)  # str -> idx
        idx_list.append(self.EOS)  # End Token 삽입
        idx_list = self.padding(idx_list, self.PAD)
        return torch.tensor(idx_list).to(device)

    def padding(self, idx_list: List[int], padding_id: int) -> List[int]:
        if len(idx_list) < self.sequence_size:
            idx_list = idx_list + [padding_id for _ in range(self.sequence_size - len(idx_list))]

        else:
            idx_list = idx_list[:self.sequence_size]

        return idx_list


class TransformerDataset(AbstractDataset):
    def __init__(self, x_path: str, y_path: str, src_voc: Any, trg_voc: Any, sequence_size: int):
        super().__init__(x_path, y_path, src_voc, trg_voc, sequence_size)
        self.PAD = src_voc['<pad>']
        self.BOS = trg_voc['<s>']
        self.EOS = trg_voc['</s>']

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        encoder_input = self.encoder_input_to_vector(self.x[idx])
        decoder_input = self.decoder_input_to_vector(self.y[idx])
        decoder_output = self.decoder_output_to_vector(self.y[idx])
        return encoder_input, decoder_input, decoder_output


class TransformerTestDataset(AbstractDataset, metaclass=ABCMeta):
    def __init__(self, x_path: str,
                 src_voc: Any,
                 sequence_size: int,
                 y_path: str = None,
                 trg_voc: Any = None,
                 ):
        super().__init__(x_path, y_path, src_voc, trg_voc, sequence_size)
        self.PAD = src_voc['<pad>']

    def __getitem__(self, idx: int) -> torch.Tensor:
        encoder_input = self.encoder_input_to_vector(self.x[idx])
        return encoder_input
