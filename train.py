import torch
import torch.nn as nn
from transformer import Transformer
from dataset import prepare_dataloader, BATCH_SIZE


MODEL_DIM = 128
HIDDEN_DIM = 512
NUM_HEADS = 4
NUM_LAYERS = 2
MAX_SEQUENCE_LENGTH = 64
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model():
    print(f"\n{'=' * 55}")
    print("  Laboratório 05 — Treinamento Fim-a-Fim do Transformer")
    print(f"{'=' * 55}")
    print(f"  Dispositivo: {DEVICE}")

    dataloader, vocab_size, pad_token_id, tokenizer = prepare_dataloader(
        subset_size=1000,
        batch_size=BATCH_SIZE,
    )

    model = Transformer(
        source_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
        model_dim=MODEL_DIM,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_length=MAX_SEQUENCE_LENGTH,
    ).to(DEVICE)

    trainable_params = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    print(f"  Parâmetros treináveis: {trainable_params:,}\n")

    loss_function = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    print(f"{'Epoch':>6} | {'Loss':>10} | {'Progresso'}")
    print("-" * 45)

    loss_history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()

        accumulated_loss = 0.0
        token_count = 0

        for source_batch, target_batch in dataloader:
            source_batch = source_batch.to(DEVICE)
            target_batch = target_batch.to(DEVICE)

            decoder_input = target_batch[:, :-1]
            decoder_target = target_batch[:, 1:]

            predictions = model(source_batch, decoder_input)

            predictions_flat = predictions.reshape(-1, predictions.size(-1))
            targets_flat = decoder_target.reshape(-1)

            loss = loss_function(predictions_flat, targets_flat)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            valid_tokens = (targets_flat != pad_token_id).sum().item()
            accumulated_loss += loss.item() * valid_tokens
            token_count += valid_tokens

        average_loss = accumulated_loss / max(token_count, 1)
        loss_history.append(average_loss)

        progress_ratio = epoch / NUM_EPOCHS
        progress_bar = "█" * int(progress_ratio * 20) + "░" * (
            20 - int(progress_ratio * 20)
        )

        print(
            f"  {epoch:>4} | {average_loss:>10.4f} | "
            f"{progress_bar} {progress_ratio * 100:.0f}%"
        )

    print("-" * 45)

    reduction = (
        (loss_history[0] - loss_history[-1]) / loss_history[0]
    ) * 100 if loss_history else 0.0

    print(f"\n  Loss inicial : {loss_history[0]:.4f}")
    print(f"  Loss final   : {loss_history[-1]:.4f}")
    print(f"  Redução total: {reduction:.1f}% ✓\n")

    torch.save(model.state_dict(), "transformer_trained.pt")
    print("  Modelo salvo em: transformer_trained.pt")

    return model, tokenizer, loss_history


if __name__ == "__main__":
    train_model()