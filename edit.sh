python ./scripts/edit.py \
    --model_type sd1 \
    --prompt_ori "a painting of a squirrel eating a burger" \
    --prompt_edit "a painting of a lion eating a burger" \
    --keyword_ori squirrel \
    --keyword_edit lion \
    --in_file ./squirrel.png \
    --out_file ./squirrel_to_lion.png \
    --guidance_scale 7.5 \
    --self_attn_span 0.4 \
    --cross_attn_span 0.8 \
    --fp16 \
    --device cuda

