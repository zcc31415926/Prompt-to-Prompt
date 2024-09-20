python ./scripts/refine.py \
    --model_type sd1 \
    --prompt_ori "a painting of a squirrel eating a burger" \
    --prompt_refine "a painting of a white squirrel eating a burger" \
    --target squirrel \
    --in_file ./squirrel.png \
    --out_file ./squirrel_to_white.png \
    --guidance_scale 7.5 \
    --self_attn_span 0.4 \
    --cross_attn_span 0.8 \
    --fp16 \
    --device cuda

