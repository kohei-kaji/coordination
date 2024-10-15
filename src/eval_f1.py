import argparse
from pathlib import Path
from PYEVALB import scorer


def main(args):
    scorer.Scorer().evalb(args.gold_path, args.test_path, args.result_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_path", type=Path, required=True)
    parser.add_argument("--test_path", type=Path, required=True)
    parser.add_argument("--result_path", type=Path, required=True)
    args = parser.parse_args()
    main(args)
