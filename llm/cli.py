from chat import main as chat_fn
from pretrain import pretrain as pretrain_fn
from finetune_lora import finetune as finetune_lora_fn
from jsonargparse import CLI

PARSER_DATA = {
    "chat": chat_fn,
    "pretrain": pretrain_fn,
    "finetune": finetune_lora_fn
}

def main():
    CLI(PARSER_DATA)

if __name__=="__main__":
    main()