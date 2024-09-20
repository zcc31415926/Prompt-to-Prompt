python ./scripts/reweight.py \
    --model_type sd1 \
    --prompt_ori "a painting of a squirrel eating a burger" \
    --prompt_reweight "a painting of a white squirrel eating a burger" \
    --target squirrel \
    --keyword white \
    --refine \
    --in_file ./squirrel.png \
    --out_file ./squirrel_to_white.png \
    --guidance_scale 7.5 \
    --emphasis_scale 3 \
    --self_attn_span 0.4 \
    --cross_attn_span 0.8 \
    --fp16 \
    --device cuda

