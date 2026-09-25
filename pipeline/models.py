"""Model/tokenizer construction, including the credential-free tiny smoke model.

The smoke profile uses a tiny randomly-initialised Gemma2 architecture with a
byte-level BPE tokenizer built in-process, so the local e2e test needs no
credentials and no network at all. (It previously fetched the gpt2 tokenizer,
whose one-time download was the gate's only network dependency — and a sandbox
egress allowlist without huggingface.co blocked the whole gate.)
"""

from pathlib import Path

TINY_SENTINEL = "tiny-random"


def _tiny_tokenizer():
    """Offline WordLevel vocabulary: the colour words plus filler, built in
    milliseconds. The tiny random model needs *a* vocab, not gpt2's — and a
    vocab this small means even random logits emit colour words, so the eval
    stage reliably has something to parse. Out-of-vocabulary prompt words map
    to <unk>, which is fine for a wiring test."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    words = ["red", "green", "blue", "yellow", "purple", "orange",
             "the", "colour", "is", "i", "choose", "."]
    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2,
             **{w: i + 3 for i, w in enumerate(words)}}
    tok = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>",
        unk_token="<unk>",
        # Gemma2 forwards reject token_type_ids, which the generic fast
        # tokenizer would otherwise emit
        model_input_names=["input_ids", "attention_mask"],
    )


def build_tokenizer(model_name: str):
    if model_name == TINY_SENTINEL:
        return _tiny_tokenizer()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def tiny_gemma2_config(vocab_size: int):
    from transformers import Gemma2Config

    return Gemma2Config(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=512,
    )


def build_causal_lm(model_name: str, dtype: str, tokenizer):
    import torch
    from transformers import AutoModelForCausalLM

    torch_dtype = getattr(torch, dtype)
    if model_name == TINY_SENTINEL:
        cfg = tiny_gemma2_config(len(tokenizer))
        return AutoModelForCausalLM.from_config(cfg)
    # Gemma 2 requires eager attention: sdpa ignores its attention logit
    # softcapping and produces NaNs in bf16.
    return AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, attn_implementation="eager"
    )


def build_seq_cls(model_name: str, dtype: str, tokenizer):
    """Reward model backbone: single-logit sequence classifier."""
    import torch
    from transformers import AutoModelForSequenceClassification, Gemma2ForSequenceClassification

    if model_name == TINY_SENTINEL:
        cfg = tiny_gemma2_config(len(tokenizer))
        cfg.num_labels = 1
        cfg.pad_token_id = tokenizer.pad_token_id
        return Gemma2ForSequenceClassification(cfg)
    torch_dtype = getattr(torch, dtype)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, torch_dtype=torch_dtype,
        attn_implementation="eager",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    return model


def lora_config(lora: dict, task_type: str):
    from peft import LoraConfig

    return LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        task_type=task_type,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )


def save_adapter(model, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
