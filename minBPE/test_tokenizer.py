import pytest
from basic import BasicTokenizer
from tokenizer_regex import RegexTokenizer

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

def test_basic_tokenizer():

    tokenizer = BasicTokenizer()

    with open("taylorswift.txt", "r") as f:
        text = f.read()

    tokenizer.train(text, 1024)
    tokenizer.save("basictokenizer")

    test_strings = [
        "", # empty string
        "?", # single character
        "hello world!!!? (안녕하세요!) lol123 😉", # fun small string
        "FILE:taylorswift.txt", # FILE: is handled as a special string in unpack()
    ]

    for text in test_strings:
        ids = tokenizer.encode(text)
        decoded_text = tokenizer.decode(ids)
        print("encoded text",text)
        print("encoded ids",ids)
        print("decoded text",decoded_text)

def test_regex_gpt2split_tokenizer():
    tokenizer = RegexTokenizer(pattern=GPT2_SPLIT_PATTERN)

    with open("taylorswift.txt", "r") as f:
        text = f.read()

    tokenizer.train(text, 1024)
    tokenizer.save("gpt2")

    test_strings = [
        "", # empty string
        "?", # single character
        "hello world!!!? (안녕하세요!) lol123 😉", # fun small string
        "FILE:taylorswift.txt", # FILE: is handled as a special string in unpack()
    ]

    for text in test_strings:
        ids = tokenizer.encode(text)
        decoded_text = tokenizer.decode(ids)
        print("encoded text",text)
        print("encoded ids",ids)
        print("decoded text",decoded_text)

def test_regex_gpt4split_tokenizer():
    tokenizer = RegexTokenizer()

    with open("taylorswift.txt", "r") as f:
        text = f.read()

    tokenizer.train(text, 1024)
    tokenizer.save("gpt4")


    test_strings = [
        "", # empty string
        "?", # single character
        "hello world!!!? (안녕하세요!) lol123 😉", # fun small string
        "FILE:taylorswift.txt", # FILE: is handled as a special string in unpack()
    ]

    for text in test_strings:
        ids = tokenizer.encode(text)
        decoded_text = tokenizer.decode(ids)
        print("encoded text",text)
        print("encoded ids",ids)
        print("decoded text",decoded_text)

if __name__=="__main__":
    pytest.main()