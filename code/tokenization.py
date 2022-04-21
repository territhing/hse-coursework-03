import enum
import json
import pathlib
import pprint
import typing as tp

import pandas as pd
import tokenizers
import torch
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from transformers import AutoTokenizer, BertTokenizerFast

TOKENIZER_CONFIG_PATH = pathlib.Path("data/punctuation-tokenizer.json")


class SpecialTokens(enum.Enum):
    CLS = "[CLS]"
    SEP = "[SEP]"
    EMPTY = "[EMPTY]"
    UNK = "[UNK]"


class TokenizationError(Exception):
    pass


class TokenizationEmptyVocabularyError(TokenizationError):
    pass


def _is_word(token: str) -> bool:
    return all(s.isalnum() or s == "" for s in token.split("_"))


class PunctuationTokenizer:
    def __init__(self, model_name: str, truncation_threshold: tp.Optional[int] = None):
        self.backend_tokenizer: BertTokenizerFast = AutoTokenizer.from_pretrained(
            model_name
        )
        self.punctuation_tokenizer = pre_tokenizers.Split(
            tokenizers.Regex(r"\w+"), behavior="isolated"
        )
        self.punctuation_vocab: tp.Optional[tp.Dict[str, int]] = None
        self.inv_punctuation_vocab: tp.Optional[tp.Dict[int, str]] = None
        self.truncation_threshold = truncation_threshold

    def encode(self, text: str) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        if self.punctuation_vocab is None:
            raise TokenizationEmptyVocabularyError(
                "called tokenizer with an empty vocabulary. Consider calling tokenizer.build_vocab() beforehand"
            )
        if not text:
            return torch.empty(1), torch.empty(1)
        if self.truncation_threshold is not None:
            text = text[: self.truncation_threshold]
        tokens = [
            token for token, _ in self.punctuation_tokenizer.pre_tokenize_str(text)
        ]
        if not _is_word(tokens[0]):
            # TODO: What to do with the first punctuation symbol?
            tokens.pop(0)
        if not tokens:
            return torch.empty(1), torch.empty(1)
        if _is_word(tokens[-1]):
            tokens.append(SpecialTokens.EMPTY.value)

        # Prepend the [CLS] token, which is required for BERT-like models
        words = [self.backend_tokenizer.vocab[SpecialTokens.CLS.value]]
        punctuations = [self.punctuation_vocab[SpecialTokens.CLS.value]]

        # Concatenate (word, punctuation-symbol) pairs with an appropriate tokenization of each
        for word, punc in zip(tokens[::2], tokens[1::2]):
            # Encode single word and strip [CLS] and [SEP] from its boundaries
            word_tokenized = self.backend_tokenizer.encode(word)[1:-1]
            if not word_tokenized:
                continue
            punc_tokenized = [self.punctuation_vocab[SpecialTokens.EMPTY.value]] * len(
                word_tokenized
            )
            punc_tokenized[-1] = (
                self.punctuation_vocab[punc]
                if punc in self.punctuation_vocab
                else self.punctuation_vocab[SpecialTokens.EMPTY.value]
            )
            words.extend(word_tokenized)
            punctuations.extend(punc_tokenized)

        # Append the [SEP] token, which is required for BERT-like models
        words.append(self.backend_tokenizer.vocab[SpecialTokens.SEP.value])
        punctuations.append(self.punctuation_vocab[SpecialTokens.SEP.value])

        inputs = torch.tensor(words)
        targets = torch.tensor(punctuations)
        attention_mask = torch.ones_like(inputs)
        return inputs, attention_mask, targets

    def decode(
        self, inputs: torch.Tensor, attention_mask: torch.Tensor, targets: torch.Tensor
    ) -> str:
        # Compute sequence length without a [SEP] token
        sequence_length = attention_mask.sum() - 1
        text = ""

        for input_id, target_id, attention_bit in zip(
            inputs[1:sequence_length].tolist(),
            targets[1:sequence_length].tolist(),
            attention_mask[1:sequence_length].tolist(),
        ):
            if attention_bit == 0:
                continue
            word_token = self.backend_tokenizer.decode(input_id)
            word_token = (
                word_token if not word_token.startswith("##") else word_token[2:]
            )
            punc_token = self.inv_punctuation_vocab[target_id]
            punc_token = (
                ""
                if punc_token in [SpecialTokens.EMPTY.value, SpecialTokens.UNK.value]
                else punc_token
            )
            text += word_token + punc_token

        return text

    def build_vocab(
        self,
        data: tp.Sequence[str],
        filename: pathlib.Path,
        *,
        vocab_size: int = 30000,
        min_frequency: int = 1000,
    ) -> None:
        tokenizer = Tokenizer(WordLevel(unk_token=SpecialTokens.UNK.value))  # type: ignore
        tokenizer.pre_tokenizer = pre_tokenizers.Split(
            tokenizers.Regex(r"\w+"), behavior="removed"
        )
        trainer = WordLevelTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=[t.value for t in SpecialTokens],
        )
        tokenizer.train_from_iterator(data, trainer=trainer)
        self.punctuation_vocab = tokenizer.get_vocab()
        self.inv_punctuation_vocab = {
            value: key for key, value in self.punctuation_vocab.items()
        }
        with open(filename, "w") as fd:
            json.dump(self.punctuation_vocab, fd)

    def load_vocab(self, filename: pathlib.Path) -> None:
        if not filename.exists():
            raise FileNotFoundError(f"could not open {filename}")
        with open(filename, "r") as fd:
            self.punctuation_vocab = json.load(fd)
            self.inv_punctuation_vocab = {
                value: key for key, value in self.punctuation_vocab.items()
            }


def main():
    corpus = pd.read_csv("data/lenta-ru-news.csv")["text"].dropna()
    tokenizer = PunctuationTokenizer("cointegrated/rubert-tiny")
    cache_file_path = pathlib.Path(".") / "tokenizer_vocab.json"
    # tokenizer.build_vocab(corpus, cache_file_path)
    tokenizer.load_vocab(cache_file_path)
    for i, text in enumerate(corpus):
        inputs, attention_mask, targets = tokenizer.encode(text)
        print(tokenizer.decode(inputs, attention_mask, targets))
        if i == 9: break
    pprint.pprint(tokenizer.punctuation_vocab)
    print(len(tokenizer.punctuation_vocab))


if __name__ == "__main__":
    main()
