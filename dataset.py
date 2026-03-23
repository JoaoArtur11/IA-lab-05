from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


TOKENIZER_MODEL_NAME = "bert-base-multilingual-cased"
MAX_SEQUENCE_LENGTH = 64
DATASET_SAMPLE_SIZE = 1000
DEFAULT_BATCH_SIZE = 16
DEFAULT_PAD_TOKEN_ID = 0


def initialize_tokenizer():
    print(f"Loading tokenizer: {TOKENIZER_MODEL_NAME}")
    return AutoTokenizer.from_pretrained(TOKENIZER_MODEL_NAME)


def load_dataset_subset(sample_size=DATASET_SAMPLE_SIZE):
    print("Loading dataset opus_books (en-pt)...")
    dataset = load_dataset("Helsinki-NLP/opus_books", "en-pt", split="train")

    dataset = dataset.select(range(min(sample_size, len(dataset))))
    print(f"Loaded subset: {len(dataset)} sentence pairs")

    return dataset


def encode_translation_pair(tokenizer, source_text, target_text, max_length=MAX_SEQUENCE_LENGTH):
    source_ids = tokenizer.encode(
        source_text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )

    target_ids = tokenizer.encode(
        target_text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )

    return source_ids, target_ids


class TranslationDataset(Dataset):

    def __init__(self, raw_dataset, tokenizer, max_length=MAX_SEQUENCE_LENGTH):
        self.samples = []
        print("Tokenizing sentence pairs...")

        for record in raw_dataset:
            source_text = record["translation"]["en"]
            target_text = record["translation"]["pt"]

            source_ids, target_ids = encode_translation_pair(
                tokenizer, source_text, target_text, max_length
            )

            if len(source_ids) < 2 or len(target_ids) < 2:
                continue

            self.samples.append(
                (
                    torch.tensor(source_ids, dtype=torch.long),
                    torch.tensor(target_ids, dtype=torch.long),
                )
            )

        print(f"Valid pairs after tokenization: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def batch_collate_fn(batch):
    source_batch, target_batch = zip(*batch)

    source_padded = pad_sequence(
        source_batch, batch_first=True, padding_value=DEFAULT_PAD_TOKEN_ID
    )
    target_padded = pad_sequence(
        target_batch, batch_first=True, padding_value=DEFAULT_PAD_TOKEN_ID
    )

    return source_padded, target_padded


def create_dataloader(sample_size=DATASET_SAMPLE_SIZE, batch_size=DEFAULT_BATCH_SIZE):
    tokenizer = initialize_tokenizer()
    raw_dataset = load_dataset_subset(sample_size)
    dataset = TranslationDataset(raw_dataset, tokenizer)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=batch_collate_fn,
    )

    vocabulary_size = tokenizer.vocab_size
    pad_token_id = tokenizer.pad_token_id or DEFAULT_PAD_TOKEN_ID

    print(f"Vocab size: {vocabulary_size} | PAD token id: {pad_token_id}")

    return dataloader, vocabulary_size, pad_token_id, tokenizer