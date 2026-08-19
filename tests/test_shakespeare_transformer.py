from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
optim = pytest.importorskip("mlx.optimizers")

from examples.shakespeare_transformer import (
    CharCorpus,
    ShakespeareTransformer,
    TransformerConfig,
    generate_text,
    parameter_count,
    save_checkpoint,
    train_model,
)


def tiny_corpus() -> CharCorpus:
    text = "To be, or not to be.\n" * 30
    vocab = tuple(sorted(set(text)))
    lookup = {char: index for index, char in enumerate(vocab)}
    tokens = np.array([lookup[char] for char in text], dtype=np.int32)
    return CharCorpus(vocab, tokens[:500], tokens[500:])


def test_default_model_is_under_ten_million_parameters():
    model = ShakespeareTransformer(80, TransformerConfig())
    assert parameter_count(model) < 10_000_000
    logits = model(mx.zeros((2, 16), dtype=mx.int32))
    assert logits.shape == (2, 16, 80)


def test_tiny_train_generate_and_save(tmp_path: Path):
    corpus = tiny_corpus()
    config = TransformerConfig(
        context_length=8, model_dim=32, num_heads=4, num_layers=1, mlp_dim=64,
    )
    model = ShakespeareTransformer(corpus.vocab_size, config)
    optimizer = optim.AdamW(learning_rate=1e-3)
    state = train_model(
        model,
        optimizer,
        corpus,
        steps=1,
        batch_size=2,
        report_every=1,
        evaluate_every=1,
        evaluation_batches=1,
    )
    assert state.step == 1
    assert np.isfinite(state.loss)
    assert np.isfinite(state.validation_loss)

    generated = generate_text(model, corpus, "To be", length=5, temperature=0)
    assert generated.startswith("To be")
    assert len(generated) == 10

    weights = save_checkpoint(model, corpus, tmp_path)
    assert weights.is_file()
    assert (tmp_path / "metadata.json").is_file()
