python run.py \
    --output_dir=./saved_models \
    --language_type=cpp \
    --train_data_file=../data/cpp/train.jsonl \
    --eval_data_file=../data/cpp/valid.jsonl \
    --test_data_file=../data/cpp/test.jsonl \
    --model_name_or_path=../../huggingface_models/microsoft/codebert-base/ \
    --epoch 20 \
    --block_size 400 \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --gradient_accumulation_steps 8 \
    --do_train \
    --evaluate_during_training \
    --seed 123456 2>&1 | tee ../log/train.log