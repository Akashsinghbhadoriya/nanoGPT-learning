from base import get_stats, merge, Tokenizer
import regex as re

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(Tokenizer):

    def __init__(self, pattern=None):
        
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, text, vocab_size):

        assert vocab_size >= 256, "vocab size should be greater than 256"
        num_ranges = vocab_size - 256
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = [list(chunk.encode("utf-8")) for chunk in text_chunks]
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_ranges):

            stats = {}
            for chunk in ids:
                get_stats(chunk, stats)
            
            pair = max(stats, key=stats.get)
            idx = i + 256
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]

        self.vocab = vocab
        self.merges = merges

    def register_special_token(self, special_tokens):
        
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v:k for k, v in special_tokens.items()}

    def decode(self, ids):

        part_bytes = []

        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"{idx} is an invalid token id")
        
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text
    
    def _encode_chunk(self, text_bytes):

        ids = list(text_bytes)

        while len(ids) >= 2:
            stats = get_stats(ids)

            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        
        return ids
    
    def ordinary_encode(self, text):

        text_chunks = re.findall(self.compiled_pattern, text)

        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)

        return ids
    
    def encode(self, text, allowed_special="none_raise"):
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        """
        special = {}
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        
        if not special:
            return self.ordinary_encode(text)
        
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.findall(special_pattern, text)

        ids = []
        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.ordinary_encode(part))
        return ids
    