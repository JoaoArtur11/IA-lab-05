import torch
from transformer import Transformer
from dataset import prepare_dataloader, BATCH_SIZE, MAX_LEN, tokenize_pair

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def greedy_autoregressive_decode(
    model,
    source_token_ids,
    tokenizer,
    max_sequence_length: int = MAX_LEN,
):
    model.eval()

    with torch.no_grad():
        source_tensor = torch.tensor(
            source_token_ids, dtype=torch.long
        ).unsqueeze(0).to(DEVICE)

        encoder_outputs = model.encode(source_tensor)

        start_token_id = tokenizer.cls_token_id
        end_token_id = tokenizer.sep_token_id

        generated_ids = [start_token_id]

        for _ in range(max_sequence_length):
            target_tensor = torch.tensor(
                generated_ids, dtype=torch.long
            ).unsqueeze(0).to(DEVICE)

            decoder_outputs = model.decode(target_tensor, encoder_outputs)

            logits = model.output_projection(decoder_outputs[:, -1, :])

            next_token_id = logits.argmax(dim=-1).item()
            generated_ids.append(next_token_id)

            if next_token_id == end_token_id:
                break

    return generated_ids


def run_overfitting_evaluation(
    model,
    tokenizer,
    source_sentence: str,
    expected_translation: str,
):
    print(f"\n{'=' * 55}")
    print("  Tarefa 4 — Prova de Fogo (Overfitting Test)")
    print(f"{'=' * 55}")
    print(f"  Frase fonte (EN) : {source_sentence}")
    print(f"  Tradução esperada: {expected_translation}")

    source_ids, _ = tokenize_pair(tokenizer, source_sentence, expected_translation)

    predicted_ids = greedy_autoregressive_decode(
        model, source_ids, tokenizer
    )

    predicted_text = tokenizer.decode(
        predicted_ids, skip_special_tokens=True
    )

    print(f"  Tradução gerada  : {predicted_text}")
    print(f"\n  IDs gerados: {predicted_ids}")
    print(f"{'=' * 55}\n")

    return predicted_text


if __name__ == "__main__":
    from train import train_model, MODEL_DIM, HIDDEN_DIM, NUM_HEADS, NUM_LAYERS

    model, tokenizer, loss_history = train_model()

    from datasets import load_dataset

    dataset = load_dataset("Helsinki-NLP/opus_books", "en-pt", split="train")
    dataset = dataset.select(range(1000))

    sample = dataset[0]["translation"]
    source_sentence = sample["en"]
    expected_translation = sample["pt"]

    run_overfitting_evaluation(
        model,
        tokenizer,
        source_sentence,
        expected_translation,
    )

    print("\n--- Mais exemplos do conjunto de treino ---")

    for index in [1, 2, 3]:
        sample = dataset[index]["translation"]

        print(f"\nFrase {index + 1} (EN): {sample['en'][:60]}...")

        run_overfitting_evaluation(
            model,
            tokenizer,
            sample["en"],
            sample["pt"],
        )