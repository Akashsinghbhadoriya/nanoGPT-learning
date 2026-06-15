
from base import get_stats, merge, Tokenizer

class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size):

        num_merges = vocab_size - 256
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        
        for i in range(num_merges):

            stat = get_stats(ids)
            pair = max(stat, key=stat.get)
            idx = 256 + i
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            ids = merge(ids, pair, idx)
        
        self.merges = merges
        self.vocab = vocab

    def decode(self, ids):

        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):

        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        while len(ids) >= 2:

            stat = get_stats(ids)
            pair = min(stat, key=lambda p: self.merges.get(p, float("inf"))) # take the pair which occurs minimum times

            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids
    
