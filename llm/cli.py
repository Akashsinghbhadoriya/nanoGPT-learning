from chat import main as chat_fn
from jsonargparse import CLI

PARSER_DATA = {
    "chat": chat_fn
}

def main():
    CLI(PARSER_DATA)

if __name__=="__main__":
    main()