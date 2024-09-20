import torch
import argparse
from PIL import Image

from unified_pipeline import SD
from prompt_to_prompt import edit, PromptToPromptPipeline
from unified_pipeline.config import getModelCfg, getGuidanceScale

torch.set_grad_enabled(False)


parser = argparse.ArgumentParser(description='')
parser.add_argument('--model_type', type=str, default='sd1',
                    choices=['sd1', 'sdxl'], help='diffusion backbone')
parser.add_argument('--prompt_ori', type=str, default='',
                    help='prompt as description of the original image')
parser.add_argument('--prompt_edit', type=str, default='',
                    help='prompt as condition for guided editing')
parser.add_argument('--keyword_ori', type=str, default='',
                    help='keyword in `prompt_ori` to be edited')
parser.add_argument('--keyword_edit', type=str, default='',
                    help='keyword in `prompt_edit` ' +
                         'as editing instructions')
parser.add_argument('--in_file', type=str, required=True,
                    help='path to the original image to be edited')
parser.add_argument('--out_file', type=str, default='edited.png',
                    help='path for saving the edited image')
parser.add_argument('--guidance_scale', type=float,
                    help='guidance scale in classifier-free guidance')
parser.add_argument('--self_attn_span', type=float, default=0.4,
                    help='step span of `keyword_ori` ' +
                         'self-attention in the process')
parser.add_argument('--cross_attn_span', type=float, default=0.8,
                    help='step span of `keyword_edit` ' +
                         'cross-attention in the process')
parser.add_argument('--fp16', action='store_true',
                    help='whether to use FP16 precision')
parser.add_argument('--device', type=str, default='cuda',
                    help='device to load the model and the data')
args = parser.parse_args()

# initialization
model_config = getModelCfg(args.model_type, True, args.fp16, args.device)
sd = SD(**model_config)
pipeline = PromptToPromptPipeline(sd)

guidance_scale = getGuidanceScale(args.model_type) \
    if args.guidance_scale is None else args.guidance_scale
print(f'Original prompt: {args.prompt_ori}')
print(f'Editing prompt: {args.prompt_edit}')
print(f'Keyword of the original prompt: {args.keyword_ori}')
print(f'Keyword of the editing prompt: {args.keyword_edit}')

# editing
img = Image.open(args.in_file).convert('RGB')
img = sd.transforms(img).unsqueeze(0).to(sd.device)
z0 = sd.image2Latent(img)
zT, inv_offsets = pipeline.getInvOffsets(
    z0, args.prompt_ori, guidance_scale)
img_edited = edit(pipeline, zT, [args.prompt_ori, args.prompt_edit],
                  [args.keyword_ori, args.keyword_edit], guidance_scale,
                  inv_offsets, args.self_attn_span, args.cross_attn_span)
Image.fromarray(img_edited).save(args.out_file)

