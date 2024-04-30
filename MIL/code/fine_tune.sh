python run.py \
    --output_dir=./saved_models/functions_loss_conv_16/ \
    --model_name_or_path=../../huggingface_models/microsoft/unixcoder-base/ \
    --language_type=java \
    --train_data_file=../data/java/train.jsonl \
    --eval_data_file=../data/java/valid.jsonl \
    --test_data_file=../data/java/test.jsonl \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --gradient_accumulation_steps 16 \
    --do_train \
    --evaluate_during_training \
    --seed 123456