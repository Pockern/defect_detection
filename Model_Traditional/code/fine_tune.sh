python run.py \
    --output_dir=./saved_models \
    --model_name_or_path=../../huggingface_models/microsoft/codebert-base/ \
    --language_type=c \
    --train_data_file=../../MIL/data/c/train.jsonl \
    --eval_data_file=../../MIL/data/c/valid.jsonl \
    --test_data_file=../../MIL/data/c/test.jsonl \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 8 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --do_train \
    --evaluate_during_training \
    --seed 123456

python run.py \
    --output_dir=./saved_models \
    --model_name_or_path=../../huggingface_models/microsoft/codebert-base/ \
    --language_type=java \
    --train_data_file=../../MIL/data/java/train.jsonl \
    --eval_data_file=../../MIL/data/java/valid.jsonl \
    --test_data_file=../../MIL/data/java/test.jsonl \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 8 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --do_train \
    --evaluate_during_training \
    --seed 123456

python run.py \
    --output_dir=./saved_models \
    --model_name_or_path=../../huggingface_models/microsoft/codebert-base/ \
    --language_type=py \
    --train_data_file=../../MIL/data/py/train.jsonl \
    --eval_data_file=../../MIL/data/py/valid.jsonl \
    --test_data_file=../../MIL/data/py/test.jsonl \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 8 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --do_train \
    --evaluate_during_training \
    --seed 123456

python run.py \
    --output_dir=./saved_models \
    --model_name_or_path=../../huggingface_models/microsoft/codebert-base/ \
    --language_type=js \
    --train_data_file=../../MIL/data/js/train.jsonl \
    --eval_data_file=../../MIL/data/js/valid.jsonl \
    --test_data_file=../../MIL/data/js/test.jsonl \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 8 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --do_train \
    --evaluate_during_training \
    --seed 123456