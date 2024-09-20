# Prompt-to-Prompt

Slight modifications on the original [Prompt-to-Prompt](https://github.com/google/prompt-to-prompt) repo to make it a portable library

## Overview

This repo is a modified version of the original [Prompt-to-Prompt](https://github.com/google/prompt-to-prompt) repo

The directory `prompt_to_prompt` in this repo is a portable library with APIs: `edit`, `refine` and `reweight` in the original paper, together with a custom `PromptToPromptPipeline` pipeline

This repo also implements a basic version of [PnP-Inversion](https://arxiv.org/abs/2310.01506) as an advanced inversion technique for editing

## Supported Pipelines

### [diffusers.StableDiffusionPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline)

- [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4)
- runwayml/stable-diffusion-v1-5 (deprecated)

### [diffusers.StableDiffusionXLPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLPipeline)

- [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) with [stabilityai/stable-diffusion-xl-refiner-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0)

> [PNDMScheduler](https://huggingface.co/docs/diffusers/api/schedulers/pndm#diffusers.PNDMScheduler), [EulerDiscreteScheduler](https://huggingface.co/docs/diffusers/api/schedulers/euler#diffusers.EulerDiscreteScheduler) and [FlowMatchEulerDiscreteScheduler](https://huggingface.co/docs/diffusers/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler) in [*diffusers*](https://github.com/huggingface/diffusers) do not support accurate inversion

## Usage

> This repo requires the [Unified-SD-Pipeline](https://github.com/zcc31415926/Unified-SD-Pipeline) repo

1. Run `git clone https://github.com/zcc31415926/Unified-SD-Pipeline.git` to clone the foundation pipeline repo

1. Run `git clone https://github.com/zcc31415926/Prompt-to-Prompt.git` to clone the prompt-to-prompt repo

1. Copy all contents in `Prompt-to-Prompt` into `Unified-SD-Pipeline`, and merge the two `scripts` directories

Run `./edit.sh`, `refine.sh` or `reweight.sh` to edit / refine / reweight a given image according to given settings

Modify `./edit.sh`, `refine.sh` or `reweight.sh` for customized experimental settings

