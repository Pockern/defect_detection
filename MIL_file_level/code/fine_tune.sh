python run.py \
    --output_dir=./saved_models \
    --language_type=py \
    --train_data_file=../data/py/train.jsonl \
    --eval_data_file=../data/py/valid.jsonl \
    --test_data_file=../data/py/test.jsonl \
    --model_name_or_path=../../huggingface_models/microsoft/codebert-base/ \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --do_train \
    --evaluate_during_training \
    --seed 123456 2>&1 | tee ../log/train.log