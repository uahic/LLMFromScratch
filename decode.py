import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import torch
from torch import Tensor
from torch.distributions import Categorical
from jaxtyping import Bool, Float, Int

from cs336_basics.activation_fcn import softmax
from cs336_basics.transformer import TransformerLM
from cs336_basics.tokenizer import load_tokenizer, BPETokenizer, STOP_TOKEN_ID
from cs336_basics.checkpoint import load_checkpoint, find_most_recent_checkpoint
from cs336_basics.config import load_model_config, ModelConfig


def decode_output(
    x: Int[Tensor, " sequence_length"],
    model: TransformerLM,
    eos_token_id: int,
    max_seq_len: int,
    p: float = 1.0,
    temperature: float | None = None,
):
    assert p <= 1.0 and p > 0.0, "top p must be in the range (0,1]"
    in_seq = x.clone()
    for i in range(in_seq.shape[0], max_seq_len):

        # Model prediction
        logits = model(in_seq)

        # Use only the last sequence next-token distribution
        logits = logits[-1]

        # Sort descending logits
        logits, indices = torch.sort(logits, descending=True)

        # Convert into probabilities
        probs = softmax(logits, temperature=temperature)

        # Compute cumsum (descending as its already sorted this way)
        probs_cumsum = torch.cumsum(probs, dim=-1)

        # Top-p / Nucleus Sampling
        nucleus_mask = probs_cumsum <= p

        # Keep at least one token, in case result is empty
        nucleus_mask[0] = True

        filtered_probs = probs[nucleus_mask]
        filtered_indices = indices[nucleus_mask]

        # Renormalize
        filtered_probs = filtered_probs / filtered_probs.sum()

        # Sampling
        sample_idx = Categorical(filtered_probs).sample()
        next_idx = filtered_indices[sample_idx]

        # End of text?
        if next_idx.item() == eos_token_id:
            break

        # Concat next token
        tmp_in_seq = torch.empty(
            (in_seq.shape[0] + 1,),
            device=in_seq.device,
            dtype=in_seq.dtype,
        )

        tmp_in_seq[:-1] = in_seq
        tmp_in_seq[-1] = next_idx

        in_seq = tmp_in_seq

    return in_seq


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Traing Loop")
    parser.add_argument("prompt", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--tokenizer_file", type=str, default="./vocab.pickle")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--device", type=str)
    args = parser.parse_args()

    train_config, opt_config, dl_config, model_config = load_model_config(
        os.path.join(args.model_path, "config.json")
    )

    tokenizer = load_tokenizer(args.tokenizer_file)

    tokens = tokenizer.encode(args.prompt)
    tokens = torch.tensor(tokens, device=args.device)
    model = TransformerLM(
        model_config.vocab_size,
        model_config.context_length,
        model_config.num_layers,
        model_config.d_model,
        model_config.num_heads,
        model_config.d_ff,
        model_config.rope_theta,
        device=model_config.device,
    )
    ckpth_file = find_most_recent_checkpoint(args.model_path)
    obj = torch.load(ckpth_file)
    model.load_state_dict(obj["model_state"])
    model.eval()

    token_out = decode_output(
        tokens,
        model,
        STOP_TOKEN_ID,
        model_config.context_length,
        p=args.top_p,
        temperature=args.temperature,
    )
    response = tokenizer.decode(token_out.tolist())
    print(response)
