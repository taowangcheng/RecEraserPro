from pprint import pprint

def color_print(words: str):
    print(f"\033[0;30;43m{words}\033[0m")

def pretty_print(object, stream=None, indent=1, width=80, depth=None, compact=False, sort_dicts=True):
    pprint(object, stream=stream, indent=indent, width=width, depth=depth, compact=compact, sort_dicts=sort_dicts)
