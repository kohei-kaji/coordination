#!/bin/bash

DATA_DIR="../data/permuted_splits"
MODEL_DIR="../data/models"
RNNG_DIR="../src/rnng-pytorch"
RESULT_DIR="../result"
RESULT_DIRS=("surprisal" "parse" "f1")

STRUCTURE_TYPES=("no-reduction" "structure-reduction" "linear-reduction")
WORD_ORDERS=(
    "000000" "000001" "000010" "000011" "000100" "000101" "000110" "000111" "001000" "001001" "001010" "001011" "001100" "001101" "001110" "001111"
    "010000" "010001" "010010" "010011" "010100" "010101" "010110" "010111" "011000" "011001" "011010" "011011" "011100" "011101" "011110" "011111" 
    "100000" "100001" "100010" "100011" "100100" "100101" "100110" "100111" "101000" "101001" "101010" "101011" "101100" "101101" "101110" "101111"
    "110000" "110001" "110010" "110011" "110100" "110101" "110110" "110111" "111000" "111001" "111010" "111011" "111100" "111101" "111110" "111111"
)
SEEDS=(3435 3436 3437)

mkdir -p $DATA_DIR $MODEL_DIR $RNNG_DIR $RESULT_DIR
for word_order in "${WORD_ORDERS[@]}"; do
    for structure_type in "${STRUCTURE_TYPES[@]}"; do
        mkdir -p "$DATA_DIR/$word_order/$structure_type/cc"
        mkdir -p "$MODEL_DIR/$word_order/$structure_type/cc"
        for result_sub_dir in "${RESULT_DIRS[@]}"; do
            mkdir -p "$RESULT_DIR/$result_sub_dir/$word_order/$structure_type/cc"
        done
    done
done

for word_order in "${WORD_ORDERES[@]}"; do
    for structure_type in "${STRUCTURE_TYPES[@]}"; do
        python $RNNG_DIR/preprocess.py \
            --trainfile "$DATA_DIR/$word_order/$structure_type/trn.tree" \
            --valfile "$DATA_DIR/$word_order/$structure_type/dev.tree" \
            --testfile "$DATA_DIR/$word_order/$structure_type/tst.tree" \
            --outputfile "$DATA_DIR/$word_order/$structure_type/ptb0" \
            --vocabminfreq 2 \
            --unkmethod "berkeleyrule"
        for seed in "${SEEDS[@]}"; do
            python $RNNG_DIR/train.py \
                --train_file "$DATA_DIR/$word_order/$structure_type/ptb0-train.json" \
                --val_file "$DATA_DIR/$word_order/$structure_type/ptb0-val.json" \
                --w_dim 256 \
                --h_dim 256 \
                --num_layers 2 \
                --save_path "$MODEL_DIR/$word_order/$structure_type/cc/${seed}.pt" \
                --batch_size 128 \
                --fixed_stack \
                --strategy in_order \
                --dropout 0.3 \
                --optimizer adam \
                --num_epochs 10 \
                --lr 0.001 \
                --device cuda \
                --seed ${seed} \
                --gpu 2

            python $RNNG_DIR/beam_search.py \
                --test_file "$DATA_DIR/$word_order/$structure_type/tst.token" \
                --model_file "$MODEL_DIR/$word_order/$structure_type/cc/${seed}.pt" \
                --batch_size 20 \
                --beam_size 100 \
                --word_beam_size 10 \
                --shift_size 5 \
                --block_size 1000 \
                --gpu 2 \
                --lm_output_file "$RESULT_DIR/surprisal/$word_order/$structure_type/cc/beam_100_${seed}.txt" > "$RESULT_DIR/parse/$word_order/$structure_type/cc/beam_100_${seed}.txt" \
                --top_n_lls_output_file "$RESULT_DIR/parse/$word_order/$structure_type/cc/beam_100_top_10_lls_${seed}.txt"
        done
    done
done
