"""A compact character Transformer for the multi-source text notebook.

The default configuration has roughly 4.8 million parameters, trains on
Apple silicon with MLX, and intentionally keeps the data pipeline and model
small enough to read in one sitting.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten


@dataclass(frozen=True)
class TransformerConfig:
    context_length: int = 128
    model_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    mlp_dim: int = 1024

    def __post_init__(self) -> None:
        if self.model_dim % self.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if self.context_length < 2:
            raise ValueError("context_length must be at least 2")


@dataclass(frozen=True)
class CharCorpus:
    vocab: Tuple[str, ...]
    train: np.ndarray
    validation: np.ndarray
    sources: Tuple[str, ...] = ()

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        lookup = {char: index for index, char in enumerate(self.vocab)}
        try:
            return [lookup[char] for char in text]
        except KeyError as exc:
            raise ValueError(f"character {exc.args[0]!r} is not in the corpus") from exc

    def decode(self, token_ids) -> str:
        return "".join(self.vocab[int(index)] for index in token_ids)


@dataclass
class TrainingState:
    step: int = 0
    loss: float = 0.0
    validation_loss: float = 0.0
    tokens_per_second: float = 0.0
    elapsed_seconds: float = 0.0


class ShakespeareTransformer(nn.Module):
    """Decoder-only character language model using causal self-attention."""

    def __init__(self, vocab_size: int, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, config.model_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.model_dim)
        self.transformer = nn.TransformerEncoder(
            config.num_layers,
            config.model_dim,
            config.num_heads,
            mlp_dims=config.mlp_dim,
            norm_first=True,
        )
        self.final_norm = nn.LayerNorm(config.model_dim)
        self.output = nn.Linear(config.model_dim, vocab_size, bias=False)

    def __call__(self, tokens):
        length = tokens.shape[1]
        if length > self.config.context_length:
            raise ValueError(
                f"sequence length {length} exceeds context length "
                f"{self.config.context_length}"
            )
        positions = mx.arange(length)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(length)
        hidden = self.transformer(hidden, mask)
        return self.output(self.final_norm(hidden))


def find_text_path(filename: str, explicit_path: Optional[Path] = None) -> Path:
    """Find a local text source from either the repo or notebook cwd."""
    candidates = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path).expanduser())
    candidates.extend([
        Path.cwd() / "data" / filename,
        Path.cwd().parent / "data" / filename,
        Path(__file__).resolve().parents[1] / "data" / filename,
    ])
    for path in candidates:
        if path.is_file():
            return path.resolve()
    tried = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        f"Text source {filename!r} not found. Copy it to data/. Tried:\n" + tried
    )


def find_shakespeare_path(explicit_path: Optional[Path] = None) -> Path:
    """Backward-compatible convenience wrapper for the original example."""
    return find_text_path("shakespeare.txt", explicit_path)


PathInput = Union[str, Path]


def load_corpus(
    paths: Union[PathInput, Sequence[PathInput]],
    validation_fraction: float = 0.1,
) -> CharCorpus:
    """Load one or more text files into one shared character vocabulary.

    Each source is split independently before the fragments are concatenated.
    This guarantees that both training and validation contain every source,
    instead of making validation an accidental slice of the final file.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if isinstance(paths, (str, Path)):
        source_paths = (Path(paths),)
    else:
        source_paths = tuple(Path(path) for path in paths)
    if not source_paths:
        raise ValueError("at least one text source is required")

    texts = []
    for path in source_paths:
        text = path.read_text(encoding="utf-8")
        if not text:
            raise ValueError(f"corpus source is empty: {path}")
        texts.append(text)

    vocab = tuple(sorted(set().union(*(set(text) for text in texts))))
    lookup = {char: index for index, char in enumerate(vocab)}
    train_parts = []
    validation_parts = []
    for path, text in zip(source_paths, texts):
        split = int(len(text) * (1.0 - validation_fraction))
        if split <= 0 or split >= len(text):
            raise ValueError(
                "validation_fraction must leave non-empty train and validation "
                f"sets for {path}"
            )
        train_parts.append(np.fromiter(
            (lookup[char] for char in text[:split]), dtype=np.int32,
        ))
        validation_parts.append(np.fromiter(
            (lookup[char] for char in text[split:]), dtype=np.int32,
        ))
    return CharCorpus(
        vocab=vocab,
        train=np.concatenate(train_parts),
        validation=np.concatenate(validation_parts),
        sources=tuple(str(path.resolve()) for path in source_paths),
    )


def random_batch(
    tokens: np.ndarray,
    batch_size: int,
    context_length: int,
    rng: np.random.Generator,
):
    max_start = len(tokens) - context_length - 1
    if max_start <= 0:
        raise ValueError("dataset is shorter than context_length + 1")
    starts = rng.integers(0, max_start, size=batch_size)
    offsets = np.arange(context_length + 1)
    batch = tokens[starts[:, None] + offsets[None, :]]
    return mx.array(batch[:, :-1]), mx.array(batch[:, 1:])


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.size for _, parameter in tree_flatten(model.parameters()))


def language_model_loss(model: ShakespeareTransformer, inputs, targets):
    logits = model(inputs)
    return nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="mean",
    )


def evaluate_loss(
    model: ShakespeareTransformer,
    tokens: np.ndarray,
    *,
    batch_size: int,
    batches: int,
    rng: np.random.Generator,
) -> float:
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(batches):
        inputs, targets = random_batch(
            tokens, batch_size, model.config.context_length, rng,
        )
        loss = language_model_loss(model, inputs, targets)
        mx.eval(loss)
        losses.append(float(loss.item()))
    if was_training:
        model.train()
    return float(np.mean(losses))


def train_model(
    model: ShakespeareTransformer,
    optimizer: optim.Optimizer,
    corpus: CharCorpus,
    *,
    steps: int,
    batch_size: int = 32,
    report_every: int = 10,
    evaluate_every: int = 100,
    evaluation_batches: int = 10,
    seed: int = 7,
    state: Optional[TrainingState] = None,
    callback: Optional[Callable[[TrainingState], None]] = None,
) -> TrainingState:
    """Train in the foreground while a Visualizer reads evaluated weights."""
    if min(steps, batch_size, report_every, evaluate_every, evaluation_batches) < 1:
        raise ValueError("training counts must all be positive")
    state = state or TrainingState()
    rng = np.random.default_rng(seed)
    loss_and_grad = nn.value_and_grad(model, language_model_loss)
    started = time.perf_counter()
    report_started = started
    report_tokens = 0

    for _ in range(steps):
        inputs, targets = random_batch(
            corpus.train, batch_size, model.config.context_length, rng,
        )
        loss, gradients = loss_and_grad(model, inputs, targets)
        optimizer.update(model, gradients)
        mx.eval(loss, model.parameters(), optimizer.state)

        state.step += 1
        state.loss = float(loss.item())
        report_tokens += batch_size * model.config.context_length
        should_report = state.step == 1 or state.step % report_every == 0
        if not should_report:
            continue

        now = time.perf_counter()
        state.elapsed_seconds += now - started
        state.tokens_per_second = report_tokens / max(now - report_started, 1e-9)
        if state.step == 1 or state.step % evaluate_every == 0:
            state.validation_loss = evaluate_loss(
                model,
                corpus.validation,
                batch_size=batch_size,
                batches=evaluation_batches,
                rng=rng,
            )
        if callback is not None:
            callback(state)
        print(
            f"step {state.step:5d} | loss {state.loss:.4f} | "
            f"val {state.validation_loss:.4f} | "
            f"{state.tokens_per_second:,.0f} tok/s"
        )
        report_started = time.perf_counter()
        started = report_started
        report_tokens = 0
    return state


def generate_text(
    model: ShakespeareTransformer,
    corpus: CharCorpus,
    prompt: str,
    *,
    length: int = 500,
    temperature: float = 0.8,
    seed: int = 7,
) -> str:
    if not prompt:
        raise ValueError("prompt must not be empty")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    token_ids = corpus.encode(prompt)
    mx.random.seed(seed)
    model.eval()
    for _ in range(length):
        context = token_ids[-model.config.context_length:]
        logits = model(mx.array(context)[None])[:, -1, :]
        if temperature == 0:
            next_token = mx.argmax(logits, axis=-1)
        else:
            next_token = mx.random.categorical(logits / temperature)
        mx.eval(next_token)
        token_ids.append(int(next_token.item()))
    return corpus.decode(token_ids)


def save_checkpoint(
    model: ShakespeareTransformer,
    corpus: CharCorpus,
    destination: Path,
) -> Path:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    weights_path = destination / "shakespeare_transformer.safetensors"
    model.save_weights(str(weights_path))
    metadata = {
        "config": asdict(model.config),
        "vocab": list(corpus.vocab),
        "sources": list(corpus.sources),
        "parameters": parameter_count(model),
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8",
    )
    return weights_path
