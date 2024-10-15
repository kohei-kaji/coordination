import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("getcwd:", os.getcwd())


RESULT_DIR: Path = Path("../result")
STRUCTURE_TYPES: list[str] = [
    "no-reduction",
    "structure-reduction",
    "linear-reduction",
]
WORD_ORDERS: list[str] = [f"{i:06b}" for i in range(64)]
SEEDS: list[str] = ["3435", "3436", "3437"]


def calculate_neg_entropy(loglikelihoods):
    # Convert loglikelihoods to probabilities
    loglikelihoods = loglikelihoods.astype(float)
    probs = np.exp(loglikelihoods)
    # Calculate negative entropy
    neg_entropy = np.sum(probs * loglikelihoods)
    return neg_entropy


with open(
    RESULT_DIR / "predictability-parseability.csv",
    "w",
) as f:
    print(
        "structure_type,word_order,seed,split,perplexity,mean_surp,f1,top_ll_per_word,top_ten_neg_entropy_per_word,top_ll_per_sent,top_ten_neg_entropy_per_sent",
        file=f,
    )
    for word_order in tqdm(WORD_ORDERS):
        for structure_type in STRUCTURE_TYPES:
            for seed in SEEDS:
                surp_path: Path = (
                    RESULT_DIR
                    / "surprisal"
                    / word_order
                    / structure_type
                    / "cc"
                    / f"beam_100_{seed}.txt"
                )
                surps: list[float] = []
                with open(surp_path, "r") as f_surp:
                    perp = None
                    for line in f_surp:
                        if line.startswith("perplexity:"):
                            perp: str = line.split(" ")[1]
                            perp: str = perp.strip()
                        elif line.startswith("---"):
                            continue
                        else:
                            line: str = line.strip()
                            surp: float = float(line.split("\t")[-2])
                            surps.append(surp)
                mean_surp: float = sum(surps) / len(surps)

                f1_path: Path = (
                    RESULT_DIR
                    / "f1"
                    / word_order
                    / structure_type
                    / "cc"
                    / f"beam_100_{seed}.txt"
                )
                with open(f1_path, "r") as f_f1:
                    f1 = None
                    for line in f_f1:
                        if line.startswith("Bracketing FMeasure"):
                            f1: str = line.split("\t")[1]
                            f1: str = f1.strip()
                if f1 is None:
                    print(f"{word_order}, {structure_type}, {seed}")

                lls_path: Path = (
                    RESULT_DIR
                    / "parse"
                    / word_order
                    / structure_type
                    / "cc"
                    / f"beam_100_top_10_lls_{seed}.txt"
                )
                with open(lls_path, "r") as file:
                    lines = file.readlines()

                # Split the data correctly
                data = []
                for line in lines:
                    parts = line.split("\t")
                    fixed_parts = parts[:5] + parts[5].strip().split(" ")
                    data.append(fixed_parts)

                columns = [
                    "sentence_id",
                    "word_index",
                    "doc_id",
                    "word",
                    "original_word",
                ] + [f"ll_{i}" for i in range(1, 11)]
                data = pd.DataFrame(data, columns=columns)

                # Replace '-inf' with a large negative number for computation
                data.replace("-inf", -1e10, inplace=True)
                data.iloc[:, 5:] = data.iloc[:, 5:].astype(float)

                # Calculate word-level entropies
                data["top_ll"] = data["ll_1"]
                data["top_ten_entropy"] = data[[f"ll_{i}" for i in range(1, 11)]].apply(
                    lambda row: calculate_neg_entropy(row.to_numpy()), axis=1
                )

                # Calculate average entropy per word
                avg_top_ll_per_word = data["top_ll"].mean()
                avg_top_ten_neg_entropy_per_word = data["top_ten_entropy"].mean()

                # Calculate sentence-level entropies as the sum of word entropies
                sentence_top_ll_sum = data.groupby("sentence_id")["top_ll"].sum()
                sentence_top_ten_neg_entropy_sum = data.groupby("sentence_id")[
                    "top_ten_entropy"
                ].sum()

                # Calculate average sentence-level entropy
                avg_top_ll_per_sent = sentence_top_ll_sum.mean()
                avg_top_ten_neg_entropy_per_sent = (
                    sentence_top_ten_neg_entropy_sum.mean()
                )

                print(
                    f"{structure_type},{word_order},{seed},cc,{perp},{mean_surp},{f1},{avg_top_ll_per_word},{avg_top_ten_neg_entropy_per_word},{avg_top_ll_per_sent},{avg_top_ten_neg_entropy_per_sent}",
                    file=f,
                )


with open(
    RESULT_DIR / "word-by-word_surp_parseloglik.csv",
    "w",
) as f:
    print(
        "structure_type,word_order,seed,split,sentpos,surp,top_ll",
        file=f,
    )
    for word_order in tqdm(WORD_ORDERS):
        for structure_type in STRUCTURE_TYPES:
            for seed in SEEDS:
                surp_path: Path = (
                    RESULT_DIR
                    / "surprisal"
                    / word_order
                    / structure_type
                    / "cc"
                    / f"beam_100_{seed}.txt"
                )
                surps: list[float] = []
                sentposs: list[str] = []
                with open(surp_path, "r") as f_surp:
                    perp = None
                    for line in f_surp:
                        if line.startswith("perplexity:"):
                            perp: str = line.split(" ")[1]
                            perp: str = perp.strip()
                        elif line.startswith("---"):
                            continue
                        else:
                            items: list[str] = line.strip().split("\t")
                            surp: float = float(items[-2])
                            sentpos: str = items[1]
                            surps.append(surp)
                            sentposs.append(sentpos)

                lls_path: Path = (
                    RESULT_DIR
                    / "parse"
                    / word_order
                    / structure_type
                    / "cc"
                    / f"beam_100_top_10_lls_{seed}.txt"
                )
                with open(lls_path, "r") as file:
                    lines = file.readlines()

                top_lls: list[float] = []
                for line in lines:
                    items: list[str] = line.strip().split("\t")
                    top_ll: float = float(items[-1].split(" ")[0])
                    top_lls.append(top_ll)

                assert len(sentposs) == len(surps) == len(top_lls)

                for sentpos, surp, top_ll in zip(sentposs, surps, top_lls):
                    print(
                        f"{structure_type},{word_order},{seed},cc,{sentpos},{surp},{top_ll}",
                        file=f,
                    )
