python run.py \
    --output_dir=./saved_models \
    --language_type=cpp \
    --train_data_file=../data/temp.jsonl \
    --eval_data_file=../data/temp.jsonl \
    --test_data_file=../data/temp.jsonl \
    --model_name_or_path=../../huggingface_models/microsoft/codebert-base/ \
    --epoch 3 \
    --block_size 400 \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --do_train \
    --evaluate_during_training \
    --seed 123456 2>&1 | tee ../log/cpp_train.log