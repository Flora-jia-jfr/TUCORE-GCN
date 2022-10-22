export BERT_BASE_DIR=pre-trained_model/BERT/uncased_L-12_H-768_A-12/

python \
    run_classifier.py \
    --do_train \
    --do_eval \
    --encoder_type BERT \
    --data_dir datasets/DailyDialog \
    --data_name DailyDialog \
    --vocab_file $BERT_BASE_DIR/vocab.txt \
    --config_file $BERT_BASE_DIR/bert_config.json \
    --init_checkpoint $BERT_BASE_DIR/pytorch_model.bin \
    --max_seq_length 512 \
    --train_batch_size 12 \
    --learning_rate 3e-5 \
    --num_train_epochs 10.0 \
    --output_dir TUCOREGCN_BERT_DailyDialog \
    --gradient_accumulation_steps 2 \
