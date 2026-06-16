from collections import Counter
import math
import json


class UnigramTokenizer:

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.SPACE_TOKEN = "\u2581"

    def save(self, path):
        data = {
            "vocab": list(self.vocab),
            "scores": self.scores,
            "piece_to_id": self.piece_to_id
        }

        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path):

        with open(path) as f:
            data = json.load(f)


        self.vocab = set(data["vocab"])

        self.scores = data["scores"]

        self.piece_to_id = data["piece_to_id"]

        self.id_to_piece = {
            int(v): k
            for k, v in self.piece_to_id.items()
        }
    
    def encode(self, text):
        ids = []
        text = (self.SPACE_TOKEN + text.replace(" ", self.SPACE_TOKEN))
        for word in text.split():

            pieces = self.viterbi(word, self.vocab, self.scores)
            for piece in pieces:
                ids.append(self.piece_to_id[piece])
        return ids

    def decode(self, ids):
        pieces = [self.id_to_piece[id] for id in ids]
        text = "".join(pieces)
        text = text.replace(self.SPACE_TOKEN, " ")
        return text.lstrip()

    def train(self, corpus):
        corpus = corpus.replace(" ",self.SPACE_TOKEN) #replace space with a unique token
        vocab, piece_freq = self.build_seed_vocab(corpus)
        scores = self.generate_scores(piece_freq)
        # iterating over the corpus till the vocab becomes equal to the vocab_size
        while len(vocab) > self.vocab_size:
            print(len(vocab))
            word_seg, word_freq = self.segment_corpus(corpus, vocab, scores) # segmentation using viterbi all the corpus words
            token_usage = self.count_token_usage(word_seg, word_freq) # update the token usage
            vocab = self.prune_vocab(vocab, token_usage) # calculate the new vocab
            print("length vocab after pruning:",len(vocab))
            print("vocab after pruning:",vocab)
            scores = self.update_scores(vocab, token_usage) # calculate the updated scores
        
        sorted_vocab = sorted(vocab)
        self.piece_to_id = {piece:idx for idx, piece in enumerate(sorted_vocab)}
        self.id_to_piece = {idx:piece for idx, piece in enumerate(sorted_vocab)}
        self.vocab = sorted_vocab
        self.scores = scores

    def viterbi(self, word, vocab, scores):
        # DP Algorithm to calculate the best segmentation from the vocab
        n = len(word)
        best_score = [float("-inf")] * (n + 1)
        best_piece = [None] * (n + 1)

        best_score[0] = 0 #empty string has score 0
        for i in range(n):
            if best_score[i] == float("-inf"):
                continue

            for piece in vocab:
                if word.startswith(piece, i):

                    j = i + len(piece)
                    new_score = best_score[i] + scores[piece]
                    if new_score > best_score[j]:
                        best_score[j] = new_score
                        best_piece[j] = piece

        pieces = []
        #backtracking
        pos = n
        while pos > 0:
            piece = best_piece[pos]
            pieces.append(piece)
            pos -= len(piece)
        pieces.reverse()
        return pieces

    def build_seed_vocab(self, corpus, max_piece_length=8, min_freq = 5):

        #frequency of each word
        word_freqs = Counter(corpus.split())
        all_chars = set()
        print(word_freqs)
        piece_freq = {}
        #iterating over each word and generating frequencies of substrings of max_length = max_piece_length
        for word in word_freqs.keys():
            # store all the characters in the word
            all_chars.update(word)
            for start in range(len(word)):
                for end in range(start+1,
                                min(start + 1 + max_piece_length,
                                    len(word) + 1 )):
                    piece = word[start:end]
                    piece_freq[piece] = piece_freq.get(piece, 0) + word_freqs[word]
        print(piece_freq)

        #filtering out piece which have less frequency
        vocab = {piece for piece, freq in piece_freq.items() if freq >= min_freq}
        vocab.update(all_chars)
        print(sorted(vocab))
        return vocab, piece_freq
    
    def generate_scores(self, piece_freq):
        # converting frequencies into probabilites for viterbi
        scores = {}
        total = sum(piece_freq.values())
        for k, v in piece_freq.items():

            scores[k] = math.log(v/total)
        return scores
    
    def segment_corpus(self, corpus, vocab, scores):
        # segmentation of each word in the corpus using viterbi algo
        word_freq = Counter(corpus.split())
        word_seg = {}
        for word, freq in word_freq.items():
            segmentation = self.viterbi(word, vocab, scores)
            word_seg[word] = segmentation
        
        return word_seg, word_freq
    
    def count_token_usage(self, word_seg, word_freq):
        # count the token usage of the word in word_seg
        token_usage = {}
        for word, pieces in word_seg.items():
            freq = word_freq[word]

            for piece in pieces:
                token_usage[piece] = token_usage.get(piece, 0) + freq

        return token_usage
    
    def prune_vocab(self, vocab, token_usage):
        #removing the pieces which are less desired or bad tokens
        keep_chars = {p for p in vocab if len(p) == 1}

        candidates = [p for p in vocab if len(p) > 1]

        candidates.sort(key=lambda p: token_usage.get(p,0))
        remove_count = int(len(vocab) * 0.1)
        candidates = candidates[remove_count:]
        return keep_chars.union(candidates)
    
    def update_scores(self, vocab, token_usage):
        # updating the scores with the pruned vocab
        scores = {}
        total = sum(token_usage.values()) + (len(vocab) - len(token_usage))
        for piece in vocab:
            usage = max(token_usage.get(piece, 1), 1)

            scores[piece] = math.log(usage)
        return scores

if __name__ == "__main__":

    with open("taylorswift.txt") as f:
        text = f.read()

    tokenizer = UnigramTokenizer(vocab_size=1000)
    # tokenizer.train(text)

    # tokenizer.save(
    #     "taylorswift_unigram.json"
    # )
    tokenizer.load("taylorswift_unigram.json")

    text = "Hi How are you!"
    encoded_text = tokenizer.encode(text)
    print(encoded_text)
    decoded_text = tokenizer.decode(encoded_text)
    print(decoded_text)