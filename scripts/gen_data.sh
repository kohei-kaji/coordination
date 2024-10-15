#!/bin/bash

DATA_GEN_DIR="../src/artificial-languages/data_gen"
DATA_DIR="../data"
TREE_PY="../src/tree.py"


python "$DATA_GEN_DIR/sample_sentences.py" \
    -g "../src/cc-grammar.txt" \
    -n 20000 \
    -O "$DATA_GEN_DIR/." \
    -b True

python "$TREE_PY" \
    --sample_sentence_file "$DATA_GEN_DIR/sample_cc-grammar.txt" \
    --permutated_output_dir "$DATA_DIR/permuted_samples/" \
    --split_dir "$DATA_DIR/permuted_splits/"
