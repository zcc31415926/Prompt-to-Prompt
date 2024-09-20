# modified from https://github.com/google/prompt-to-prompt
import numpy as np
from PIL import Image

from .ptp_main import AttentionStore, AttentionReplace, \
    AttentionRefine, AttentionReweight, \
    LocalBlend, get_equalizer, aggregate_attention
from .ptp_utils import text_under_image


def run(pipeline, latent, prompts, controller,
        guidance_scale, inv_offsets):
    assert latent == None or latent.size(0) == len(prompts)
    image = pipeline.run(latent, prompts, controller,
                         guidance_scale, inv_offsets)
    return images[-1]


# modified from function `show_cross_attention` in
# https://github.com/google/prompt-to-prompt/blob/main/prompt-to-prompt_stable.ipynb
def getCrossAttn(pipeline, latent, prompt, guidance_scale, inv_offsets):
    controller = AttentionStore()
    image = pipeline.run(latent, prompt, controller,
                         guidance_scale, inv_offsets)
    tokens = pipeline.sd.tokenizer.encode(prompt)
    decoder = pipeline.sd.tokenizer.decode
    attention_maps = aggregate_attention(
        controller, 16, ('up', 'down'), True, 0)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    return np.hstack(images)


# self-attention map replacement from step 0 to self_attn_span * pipeline.sd.num_steps
# cross-attention map replacement from step 0 to cross_attn_span * pipeline.sd.num_steps


# editing: keyword replacement
# e.g.: 'a painting of a squirrel' -> 'a painting of a lion'
# prompts: ['a painting of a squirrel', 'a painting of a lion']
# keywords: ('squirrel', 'lion')
def edit(pipeline, latent, prompts, keywords, guidance_scale,
         inv_offsets, self_attn_span, cross_attn_span):
    lb = LocalBlend(pipeline.sd.tokenizer, prompts,
                    keywords, pipeline.sd.device)
    controller = AttentionReplace(
        pipeline.sd.tokenizer, prompts, pipeline.sd.num_steps,
        {'default_': 1, keywords[1]: cross_attn_span},
        self_attn_span, lb, pipeline.sd.device)
    image = pipeline.run(latent, prompts, controller,
                         guidance_scale, inv_offsets)
    return image[-1]


# refinement: attribute addition
# e.g.: 'a painting of a squirrel' -> 'a painting of a white squirrel'
# prompts: ['a painting of a squirrel', 'a painting of a white squirrel']
# target: 'squirrel'
def refine(pipeline, latent, prompts, target, guidance_scale,
           inv_offsets, self_attn_span, cross_attn_span):
    lb = LocalBlend(pipeline.sd.tokenizer, prompts,
                    (target, target), pipeline.sd.device)
    controller = AttentionRefine(
        pipeline.sd.tokenizer, prompts, pipeline.sd.num_steps,
        cross_attn_span, self_attn_span, lb, pipeline.sd.device)
    image = pipeline.run(latent, prompts, controller,
                         guidance_scale, inv_offsets)
    return image[-1]


# reweighting: attribute emphasis / deemphasis
# e.g.: 'a bowl of soup' -> 'a bowl of soup with croutons'
# prompts: ['a bowl of soup', 'a bowl of soup with croutons']
# keyword: 'croutons'
# target: 'soup'
# negative emphasis_scale for deemphasis
def reweight(pipeline, latent, prompts, keyword, target, refine,
             emphasis_scale, guidance_scale, inv_offsets,
             self_attn_span, cross_attn_span):
    eq = get_equalizer(pipeline.sd.tokenizer, prompts[1],
                       (keyword, ), (emphasis_scale, ))
    lb = LocalBlend(pipeline.sd.tokenizer, prompts,
                    (target, target), pipeline.sd.device)
    refine_controller = AttentionRefine(
        pipeline.sd.tokenizer, prompts, pipeline.sd.num_steps,
        cross_attn_span, self_attn_span, lb, pipeline.sd.device) \
        if refine else None
    controller = AttentionReweight(
        pipeline.sd.tokenizer, prompts, pipeline.sd.num_steps,
        cross_attn_span, self_attn_span, eq, lb,
        refine_controller, pipeline.sd.device)
    image = pipeline.run(latent, prompts, controller,
                         guidance_scale, inv_offsets)
    return image[-1]

