import multiprocessing
import os
import regex as re
import pprint
import numpy as np
import numpy.lib.format as fmt

# from profiler import profile

from tqdm import tqdm
from itertools import pairwise
from utils import find_chunk_boundaries

SPECIAL_WORDS = ["<|endoftext|>", "<|padding|>"]
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}+|\s+(?!\S)|\s+]"""
SPECIAL_TOKEN_IDS = {w: 256 + i for i, w in enumerate(SPECIAL_WORDS)}
STOP_TOKEN_ID = 256 + SPECIAL_WORDS.index("<|endoftext|>")


def load_tokenizer(file: str = "./vocab.pickle"):
    import pickle

    with open(file, "rb") as f:
        return pickle.load(f)


def init_worker(queue):
    global chunk
    chunk = queue.get()


def merge_parallel(top_pair, new_token_idx):
    global chunk
    chunk, counts = merge_tokens(chunk, top_pair, new_token_idx)
    return counts


def init_vocabulary() -> dict[[bytes, ...], int]:
    vocab = {idx: bytes([idx]) for idx in range(256)}
    return vocab


def filter_special_words(chunk: bytes, special_words: list[str]):
    pattern = "|".join(re.escape(w) for w in special_words)
    return re.sub(pattern, "", chunk)


def merge_counting_dict(base_dict: dict, update_dict: dict):
    for key, value in update_dict.items():
        base_dict[key] = base_dict.get(key, 0) + value
        if base_dict[key] <= 0:
            del base_dict[key]


def count_words(words: list[str | bytes]) -> dict[[str | bytes], int]:
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    return counts


def pretokenize_parallel(args):
    chunk, special_words = args
    chunk = filter_special_words(chunk, special_words)
    return pretokenize(chunk, pattern=PAT)


def chunk_text(file: str, stop_word, special_words):
    with open(file, "rb") as f:
        import os

        max_workers = multiprocessing.cpu_count()
        target_chunk_size = 16 * 1024 * 1024  # 64 MB per chunk
        file_size = os.path.getsize(file)
        num_desired_chunks = max(max_workers, file_size // target_chunk_size)
        boundaries = find_chunk_boundaries(f, num_desired_chunks, stop_word)

        chunks = []
        print("Reading chunks...")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)

        counts = {}
        print("Pretokenizing words...(parallel)")
        with multiprocessing.Pool(processes=max_workers) as pool:
            for i in tqdm(range(0, len(chunks), max_workers), desc="Pretokenizing"):
                batch = chunks[i : i + max_workers]
                results = pool.map(
                    pretokenize_parallel, [(c, special_words) for c in batch]
                )
                for chunk_counts in results:
                    merge_counting_dict(counts, chunk_counts)

    return counts


def pretokenize(chunk: bytes, pattern=PAT):
    word_counts = {}
    matches = re.finditer(pattern, chunk)
    for match in matches:
        key = tuple(match.group().encode("utf-8"))
        word_counts[key] = word_counts.get(key, 0) + 1
    return word_counts


def count_char_pairs(word_counts: dict) -> dict:
    counts = {}
    pair_word_indices = {}
    for word, word_count in word_counts.items():
        for pair in zip(word, word[1:]):
            counts[pair] = counts.get(pair, 0) + word_count
            ii = pair_word_indices.get(pair, set())
            ii.add(word)
            pair_word_indices[pair] = ii
    return counts, pair_word_indices


def merge_pair(
    words: dict, pair: tuple, new_token_idx: int, pair_indices: dict, pair_counts: dict
):
    for old_word in list(pair_indices[pair]):
        new_word, count_deltas = merge_tokens(old_word, pair, new_token_idx)
        new_word = tuple(new_word)
        word_count = words.pop(old_word)
        words[new_word] = word_count

        for p in pairwise(old_word):
            pair_indices.get(p, set()).discard(old_word)

        for p in pairwise(new_word):
            pair_indices.setdefault(p, set()).add(new_word)

        scaled_deltas = {k: v * word_count for k, v in count_deltas.items()}
        merge_counting_dict(pair_counts, scaled_deltas)

    del pair_counts[pair]
    del pair_indices[pair]


def merge_tokens(
    tokens: list,
    pair: tuple,
    new_token: int,
):
    i = 0
    chunk_counts = {}

    new_tokens = []
    while i < len(tokens):
        # Pair found
        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            new_tokens.append(new_token)

            # update left pair
            if i > 0:
                left_pair = (tokens[i - 1], tokens[i])
                chunk_counts[left_pair] = chunk_counts.get(left_pair, 0) - 1
                new_left_pair = (new_tokens[-2], new_token)
                chunk_counts[new_left_pair] = chunk_counts.get(new_left_pair, 0) + 1

            # update right pair
            if i + 2 < len(tokens):
                right_pair = (tokens[i + 1], tokens[i + 2])
                chunk_counts[right_pair] = chunk_counts.get(right_pair, 0) - 1
                new_right_pair = (new_token, tokens[i + 2])
                chunk_counts[new_right_pair] = chunk_counts.get(new_right_pair, 0) + 1
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens, chunk_counts


class BPETokenizer:
    def __init__(self, vocab: dict = {}, merges: dict = {}):
        self.vocab = vocab
        self.merges = merges

    def store(self, file: str = "./vocab.pickle"):
        import pickle

        with open(file, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def train(
        self,
        file: str,
        max_vocab_size: int = 600,
        special_words: list[bytes] = SPECIAL_WORDS,
        stop_word: bytes = b"<|endoftext|>",
    ):
        self.vocab = init_vocabulary()

        # Add special words as token
        for i, special_word in enumerate(special_words):
            self.vocab[256 + i] = bytes(special_word.encode("utf-8"))

        # Chunk text
        word_counts = chunk_text(file, stop_word, special_words)

        num_merges = max_vocab_size - (256 + len(special_words))

        # Initial counting of character pairs within words
        print("Count character pairs...")
        pair_counts, pair_indices = count_char_pairs(word_counts)

        # Replace pairs (merge)
        for i in tqdm(range(num_merges), desc="Merging"):
            # Improvement possibility: indexed priority queue or PQ + doubly linked list
            top_pair = max(pair_counts, key=pair_counts.get)
            new_token_idx = 256 + len(special_words) + i
            self.merges[(top_pair[0], top_pair[1])] = new_token_idx
            self.vocab[new_token_idx] = (
                self.vocab[top_pair[0]] + self.vocab[top_pair[1]]
            )
            merge_pair(word_counts, top_pair, new_token_idx, pair_indices, pair_counts)

        # Merge results from chunks

        return self.vocab, self.merges

    def decode(self, ids: list[int]) -> str:
        tokens = b"".join(self.vocab[idx] for idx in ids)
        return tokens.decode("utf-8", errors="replace")

    def encode(self, text: str) -> list[int]:
        # special_token_ids = {w: 256 + i for i, w in enumerate(SPECIAL_WORDS)}

        ids = []
        special_pattern = "|".join(
            re.escape(w) for w in sorted(SPECIAL_TOKEN_IDS, key=len, reverse=True)
        )
        parts = re.split(f"({special_pattern})", text) if special_pattern else [text]

        for part in parts:
            if part in SPECIAL_TOKEN_IDS:
                ids.append(SPECIAL_TOKEN_IDS[part])
                continue
            # Handle chunks (split words)
            for match in re.finditer(PAT, part):
                tokens = list(match.group().encode("utf-8"))
                while len(tokens) >= 2:
                    best_pair = min(
                        pairwise(tokens), key=lambda p: self.merges.get(p, float("inf"))
                    )
                    if best_pair not in self.merges:
                        break
                    tokens, _ = merge_tokens(tokens, best_pair, self.merges[best_pair])
                ids.extend(tokens)

        return ids


def train_tokenizer(
    file="./vocab.pickle", train_file: str = "../data/TinyStoriesV2-GPT4-train.txt"
):
    from pathlib import Path

    if not Path(file).exists():
        tokenizer = BPETokenizer()
        vocab, counts = tokenizer.train(train_file, max_vocab_size=10000)
        # vocab, counts = tokenizer.train("../data/TinyStoriesV2-GPT4-valid.txt")
        tokenizer.store()
        print("---- Vocabulary ----")
        pprint.pprint(vocab)
        print("---- ---------- ----")
    else:
        print("vocab.pickle already exists in the given folder. (Re)move it first")


_encode_tokenizer = None


def _init_encode_worker(vocab_file):
    global _encode_tokenizer
    _encode_tokenizer = load_tokenizer(vocab_file)


def _encode_line(line):
    return _encode_tokenizer.encode(line)


def cmd_encode(args):
    dtype = np.dtype(np.int32)
    total = 0
    file_size = os.path.getsize(args.src)
    num_workers = multiprocessing.cpu_count()

    with open(args.src, "r") as src_f:
        lines = src_f.readlines()

    with (
        open(args.dst, "wb") as dst_f,
        tqdm(total=file_size, unit="B", unit_scale=True, desc="Encoding") as pbar,
        multiprocessing.Pool(num_workers, initializer=_init_encode_worker, initargs=(args.vocab,)) as pool,
    ):
        fmt.write_array_header_2_0(
            dst_f, fmt.header_data_from_array_1_0(np.empty(0, dtype=dtype))
        )
        try:
            for line, ids in zip(lines, pool.imap(_encode_line, lines, chunksize=256)):
                chunk = np.array(ids, dtype=dtype)
                chunk.tofile(dst_f)
                total += len(chunk)
                pbar.update(len(line.encode()))
        finally:
            dst_f.seek(0)
            fmt.write_array_header_2_0(
                dst_f, {"descr": dtype.str, "fortran_order": False, "shape": (total,)}
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="tokenizer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train BPE tokenizer and save vocab")
    p_train.add_argument("train_file", help="Path to training text file")
    p_train.add_argument("--vocab", default="./vocab.pickle", help="Output vocab file")

    p_encode = sub.add_parser("encode", help="Encode a text file to token ids (.npy)")
    p_encode.add_argument("src", help="Source text file")
    p_encode.add_argument("dst", help="Output .npy file")
    p_encode.add_argument("--vocab", default="./vocab.pickle", help="Vocab file to use")

    args = parser.parse_args()

    if args.cmd == "train":
        train_tokenizer(file=args.vocab, train_file=args.train_file)
    elif args.cmd == "encode":
        cmd_encode(args)
