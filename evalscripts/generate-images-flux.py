from diffusers import FluxPipeline
import torch
from PIL import Image
import pandas as pd
import argparse
import os
from safetensors.torch import load_file

torch.enable_grad(False)

def generate_images(base_model, esd_path, prompts_path, save_path,
                    device='cuda:0', torch_dtype=torch.bfloat16,
                    guidance_scale=3.5, num_inference_steps=30,
                    num_samples=1, from_case=0):
    """
    Generate images with FluxPipeline + ESD weights.
    """
    # pipeline
    pipe = FluxPipeline.from_pretrained(base_model, torch_dtype=torch_dtype).to(device)

    
    if esd_path is not None:
        print(f"[INFO] Loading ESD weights from {esd_path}")
        esd_weights = load_file(esd_path)
        pipe.transformer.load_state_dict(esd_weights, strict=False)

    
    df = pd.read_csv(prompts_path)

    
    model_name = os.path.basename(esd_path).split('.')[0] if esd_path else "flux-base"
    folder_path = f'{save_path}/{model_name}'
    os.makedirs(folder_path, exist_ok=True)

    
    for _, row in df.iterrows():
        prompt = [str(row.prompt)] * num_samples
        seed = int(row.evaluation_seed)
        case_number = int(row.case_number)

        if case_number < from_case:
            continue

        print(f"[GEN] Case {case_number} | Prompt: {row.prompt} | Seed: {seed}")

        pil_images = pipe(prompt,
                          generator=torch.Generator().manual_seed(seed),
                          num_inference_steps=num_inference_steps,
                          guidance_scale=guidance_scale).images

        for num, im in enumerate(pil_images):
            im.save(f"{folder_path}/{case_number}_{num}.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Images with Flux + ESD')
    parser.add_argument('--base_model', type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument('--esd_path', type=str, default=None)
    parser.add_argument('--prompts_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, default='outputs/flux_eval')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--guidance_scale', type=float, default=3.5)
    parser.add_argument('--from_case', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--num_inference_steps', type=int, default=30)

    args = parser.parse_args()

    generate_images(
        base_model=args.base_model,
        esd_path=args.esd_path,
        prompts_path=args.prompts_path,
        save_path=args.save_path,
        device=args.device,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        num_samples=args.num_samples,
        from_case=args.from_case
    )
