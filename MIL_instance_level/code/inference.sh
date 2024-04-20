python run.py \
    --output_dir=./saved_models \
    --model_name_or_path=../../huggingface_models/microsoft/codebert-base/ \
    --do_eval \
    --do_test \
    --language_type=cpp \
    --train_data_file=../data/cpp/train.jsonl \
    --eval_data_file=../data/cpp/valid.jsonl \
    --test_data_file=../data/cpp/test.jsonl \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1 | tee ../log/test.log