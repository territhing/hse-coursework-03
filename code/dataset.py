import typing as tp

import torch
from tokenization import PunctuationTokenizer
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class PunctuationRestorationDataset(Dataset):
    def __init__(
        self,
        data: tp.Sequence[str],
        tokenizer: PunctuationTokenizer,
    ):
        self.tokenizer = tokenizer
        self.data = data

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        return self.tokenizer.encode(self.data[idx])

    def __len__(self) -> int:
        return len(self.data)


def pad_sequence_fixed_length(
    seqs: tp.Union[tp.List[torch.Tensor], torch.Tensor],
    length: int,
    *,
    batch_first: bool = True,
    padding_value: int = 0
) -> torch.Tensor:
    seqs[0] = nn.ConstantPad1d((0, length - seqs[0].shape[0]), value=padding_value)(
        seqs[0]
    )
    return pad_sequence(seqs, batch_first=batch_first, padding_value=padding_value)


def collate_fn(
    data: tp.List[tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    inputs, attention_masks, targets = zip(*data)
    max_length = max(len(input_) for input_ in inputs)
    inputs, attention_masks, targets = (
        pad_sequence_fixed_length(list(inputs), max_length),
        pad_sequence_fixed_length(list(attention_masks), max_length),
        pad_sequence_fixed_length(list(targets), max_length),
    )
    return inputs, attention_masks, targets
