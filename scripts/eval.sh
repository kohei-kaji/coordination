#!/bin/bash

DATA_DIR="../data/permuted_splits"
PARSE_DIR="../result/parse"
RESULT_F1_DIR="../result/f1"

EVALF1_PY="../src/eval_f1.py"
ACCUM_DATA_PY="../src/accum_data.py"
PLOT_PY="../src/plot.py"

STRUCTURE_TYPES=("no-reduction" "structure-reduction" "linear-reduction")
WORD_ORDERS=(
    "000000" "000001" "000010" "000011" "000100" "000101" "000110" "000111" "001000" "001001" "001010" "001011" "001100" "001101" "001110" "001111"
    "010000" "010001" "010010" "010011" "010100" "010101" "010110" "010111" "011000" "011001" "011010" "011011" "011100" "011101" "011110" "011111"
    "100000" "100001" "100010" "100011" "100100" "100101" "100110" "100111" "101000" "101001" "101010" "101011" "101100" "101101" "101110" "101111"
    "110000" "110001" "110010" "110011" "110100" "110101" "110110" "110111" "111000" "111001" "111010" "111011" "111100" "111101" "111110" "111111"
)
SEEDS=(3435 3436 3437)


mkdir -p $RESULT_F1_DIR

for word_order in "${WORD_ORDERS[@]}"; do
    for structure_type in "${STRUCTURE_TYPES[@]}"; do
        mkdir -p "$RESULT_F1_DIR/$word_order/$structure_type/cc"
    done
done


for word_order in "${WORD_ORDERS[@]}"; do
    for structure_type in "${STRUCTURE_TYPES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            gold_path="$DATA_DIR/$word_order/$structure_type/tst.tree"
            test_path="$PARSE_DIR/$word_order/$structure_type/cc/beam_100_${seed}.txt"
            result_path="$RESULT_F1_DIR/$word_order/$structure_type/cc/beam_100_${seed}.txt"

            if [[ -f "$gold_path" && -f "$test_path" ]]; then
                python $EVALF1_PY \
                    --gold_path "$gold_path" \
                    --test_path "$test_path" \
                    --result_path "$result_path"
            else
                echo "$gold_path か $test_path が存在しない"
            fi
        done
    done
done

python $ACCUM_DATA_PY
python $PLOT_PY
