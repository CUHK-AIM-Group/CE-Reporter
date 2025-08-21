export CUDA_VISIBLE_DEVICES=0
cd ..
python llava/eval/model_vqa.py \
    --model-path CE-Reporter-llava-1.5-7b \
    --question-file ce-imgs/test_img_to_llava.jsonl \
    --answers-file ce-imgs/answer.jsonl