"""Command line: python -m study.ask "what does the left ventricle do?"  (add --claude for Claude mode)"""
import os
import sys

from .answer import answer, load_index


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--claude"]
    use_claude = "--claude" in sys.argv
    if use_claude and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY to use --claude.")
    index = load_index()
    if args:
        print(answer(" ".join(args), index, use_claude))
        return
    print("Ask a question about the notes (blank line to quit).")
    while (q := input("> ").strip()):
        print(answer(q, index, use_claude), "\n")


if __name__ == "__main__":
    main()
