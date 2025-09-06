import os 
import torch
import sys
import random
import csv
from tqdm.auto import tqdm
from safetensors.torch import save_file
from diffusers import FluxPipeline, AutoencoderTiny
from diffusers.models import FluxTransformer2DModel
from diffusers.utils import make_image_grid
import argparse
import copy

sys.path.append('.')
from utils.flux_utils import esd_flux_call
FluxPipeline.__call__ = esd_flux_call

def load_flux_models(basemodel_id="black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, device='cuda:0'):
    
    esd_transformer = FluxTransformer2DModel.from_pretrained(basemodel_id, subfolder="transformer", torch_dtype=torch_dtype).to(device)
    pipe = FluxPipeline.from_pretrained(basemodel_id, 
                                        transformer=esd_transformer,
                                        vae=None,
                                        torch_dtype=torch_dtype, 
                                        use_safetensors=True).to(device)
    
    pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taef1", torch_dtype=torch_dtype).to(device)
    
    return pipe, esd_transformer

def set_module(module, module_name, new_module):

    if isinstance(module_name, str):
        module_name = module_name.split('.')

    if len(module_name) == 1:
        return setattr(module, module_name[0], new_module)
    else:
        module = getattr(module, module_name[0])
        return set_module(module, module_name[1:], new_module)

def get_esd_trainable_parameters(esd_transformer, train_method='esd-x'):
    esd_params = []
    esd_param_names = []
    for name, module in esd_transformer.named_modules():
        if module.__class__.__name__ in ["Linear", "Conv2d", "LoRACompatibleLinear", "LoRACompatibleConv"]:
            if train_method == 'esd-x' and 'attn' in name:
                for n, p in module.named_parameters():
                    esd_param_names.append(name+'.'+n)
                    esd_params.append(p)
                    
            if train_method == 'esd-x-strict' and ('to_k' in name or 'to_v' in name) and ('attn' in name):
                for n, p in module.named_parameters():
                    esd_param_names.append(name+'.'+n)
                    esd_params.append(p)

    return esd_param_names, esd_params

def load_scenes_from_csv(csv_path, default_negative_guidance=2.0):
    """从CSV文件加载场景配置"""
    scenes = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            scene = {
                'erase_from': row['erase_from'].strip(),
                'negative_guidance': float(row.get('negative_guidance', default_negative_guidance))
            }
            scenes.append(scene)
    return scenes

if __name__=="__main__":

    parser = argparse.ArgumentParser(
                    prog = 'TrainESD for FLUX - Multi Scene',
                    description = 'Finetuning FLUX to erase concepts from multiple scenes')
    parser.add_argument('--basemodel_id', help='model id for the model (hf compatible)', type=str, required=False, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument('--erase_concept', help='concept to erase', type=str, required=True)
    
    
    parser.add_argument('--erase_from', help='single target concept to erase from', type=str, required=False, default=None)
    parser.add_argument('--scenes_csv', help='CSV file containing multiple scenes to erase from', type=str, required=False, default=None)
    
    parser.add_argument('--num_inference_steps', help='number of inference steps for diffusion model', type=int, required=False, default=28)
    parser.add_argument('--guidance_scale', help='guidance scale to run training for diffusion model', type=float, required=False, default=1)
    parser.add_argument('--inference_guidance_scale', help='guidance scale to run inference for diffusion model', type=float, required=False, default=3.5)
    parser.add_argument('--max_sequence_length', help='max_sequence_length argument for flux models (use 256 for schnell)', type=int, required=False, default=512)
    
    parser.add_argument('--train_method', help='Type of method (esd-x, esd-x-strict)', type=str, required=True)
    parser.add_argument('--iterations_per_scene', help='Number of iterations per scene', type=int, default=1400)
    parser.add_argument('--resolution', help='resolution of image to train', type=int, default=512)
    parser.add_argument('--lr', help='Learning rate', type=float, default=1e-4)
    parser.add_argument('--batchsize', help='Batchsize', type=int, default=1)
    parser.add_argument('--negative_guidance', help='Default negative guidance value', type=float, required=False, default=2.0)
    parser.add_argument('--save_path', help='Path to save model', type=str, default='esd-models/flux/')
    parser.add_argument('--device', help='cuda device to train on', type=str, required=False, default='cuda:0')
    parser.add_argument('--save_interval', help='Save checkpoint every N iterations', type=int, default=5000)

    args = parser.parse_args()

    
    if args.erase_from is None and args.scenes_csv is None:
        raise ValueError("Must provide either --erase_from or --scenes_csv")
    if args.erase_from is not None and args.scenes_csv is not None:
        raise ValueError("Cannot provide both --erase_from and --scenes_csv")

    basemodel_id = args.basemodel_id
    erase_concept = args.erase_concept
    
    
    if args.scenes_csv:
        scenes = load_scenes_from_csv(args.scenes_csv, args.negative_guidance)
        print(f"Loaded {len(scenes)} scenes from {args.scenes_csv}")
        for i, scene in enumerate(scenes):
            print(f"  Scene {i+1}: {scene['erase_from']} (negative_guidance={scene['negative_guidance']})")
    else:
        
        scenes = [{
            'erase_from': args.erase_from,
            'negative_guidance': args.negative_guidance
        }]

    num_inference_steps = args.num_inference_steps
    guidance_scale = args.guidance_scale
    inference_guidance_scale = args.inference_guidance_scale
    train_method = args.train_method
    
    
    iterations_per_scene = args.iterations_per_scene
    max_training_steps = len(scenes) * iterations_per_scene
    print(f"\nTotal training steps: {max_training_steps} ({len(scenes)} scenes × {iterations_per_scene} iterations)")
    
    batchsize = args.batchsize
    max_sequence_length = args.max_sequence_length
    height = width = args.resolution
    lr = args.lr
    if 'esd-x' not in train_method:
        lr = 1e-5
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    device = args.device
    torch_dtype = torch.bfloat16
    save_interval = args.save_interval
    
    criteria = torch.nn.MSELoss()

    
    pipe, esd_transformer = load_flux_models(basemodel_id=basemodel_id, torch_dtype=torch_dtype, device=device)
    pipe.set_progress_bar_config(disable=True)

    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    noise_scheduler_copy = copy.deepcopy(pipe.scheduler)

    
    esd_param_names, esd_params = get_esd_trainable_parameters(esd_transformer, train_method=train_method)
    optimizer = torch.optim.Adam(esd_params, lr=lr)

    esd_param_dict = {}
    for name, param in zip(esd_param_names, esd_params):
        esd_param_dict[name] = param

    base_params = copy.deepcopy(esd_params)
    base_param_dict = {}
    for name, param in zip(esd_param_names, base_params):
        base_param_dict[name] = param
        base_param_dict[name].requires_grad_(False)

    
    print("\nPre-computing embeddings for all scenes...")
    scene_embeddings = {}
    
    for scene_config in scenes:
        scene_name = scene_config['erase_from']
        print(f"  Computing embeddings for: {scene_name}")
        
        prompts = [erase_concept, scene_name, '']
        with torch.no_grad():
            prompt_embeds_all, pooled_prompt_embeds_all, text_ids = pipe.encode_prompt(
                prompts, prompt_2=prompts, max_sequence_length=max_sequence_length
            )
            
            erase_prompt_embeds, erase_from_prompt_embeds, null_prompt_embeds = prompt_embeds_all.chunk(3)
            erase_pooled_prompt_embeds, erase_from_pooled_prompt_embeds, null_pooled_prompt_embeds = pooled_prompt_embeds_all.chunk(3)
            
            scene_embeddings[scene_name] = {
                'erase_prompt_embeds': erase_prompt_embeds,
                'erase_from_prompt_embeds': erase_from_prompt_embeds,
                'null_prompt_embeds': null_prompt_embeds,
                'erase_pooled_prompt_embeds': erase_pooled_prompt_embeds,
                'erase_from_pooled_prompt_embeds': erase_from_pooled_prompt_embeds,
                'null_pooled_prompt_embeds': null_pooled_prompt_embeds,
                'text_ids': text_ids,
                'negative_guidance': scene_config['negative_guidance']
            }
    
    
    with torch.no_grad():
        model_input = pipe.vae.encode(torch.randn((1, 3, height, width)).to(torch_dtype).to(pipe.vae.device)).latents.cpu()

    
    pipe.text_encoder_2.to('cpu')
    pipe.text_encoder.to('cpu')
    pipe.vae.to('cpu')

    torch.cuda.empty_cache()
    import gc
    gc.collect()

    
    scene_counters = {scene['erase_from']: 0 for scene in scenes}
    scene_losses = {scene['erase_from']: [] for scene in scenes}
    
    
    pbar = tqdm(range(max_training_steps), desc='Training ESD Multi-Scene')
    loss_history = {}
    global_step = 0
    
    for iteration in range(max_training_steps):
        optimizer.zero_grad()
        
        
        available_scenes = [s for s in scenes if scene_counters[s['erase_from']] < iterations_per_scene]
        if not available_scenes:
            break
            
        
        current_scene = random.choice(available_scenes)
        scene_name = current_scene['erase_from']
        scene_data = scene_embeddings[scene_name]
        
        
        scene_counters[scene_name] += 1
        
        guidance = torch.tensor([guidance_scale], device=device)
        guidance = guidance.expand(batchsize)

        
        run_till_timestep = random.randint(0, num_inference_steps-1)
        timesteps = pipe.scheduler.timesteps[run_till_timestep].unsqueeze(0).to(device)
        seed = random.randint(0, 2**15)

        latent_image_ids = FluxPipeline._prepare_latent_image_ids(
            model_input.shape[0],
            model_input.shape[2] // 2,
            model_input.shape[3] // 2,
            device,
            torch_dtype,
        )
        
        
        for key, ft_module in esd_param_dict.items():
            set_module(pipe.transformer, key, ft_module)
        pipe.transformer.eval()
        
        
        with torch.no_grad():
            xt = pipe(
                prompt_embeds=scene_data['erase_from_prompt_embeds'],
                pooled_prompt_embeds=scene_data['erase_from_pooled_prompt_embeds'],
                num_images_per_prompt=batchsize,
                num_inference_steps=num_inference_steps,
                guidance_scale=inference_guidance_scale,
                run_till_timestep=run_till_timestep,
                generator=torch.Generator().manual_seed(seed),
                output_type='latent',
                height=height,
                width=width,
            ).images
        
        
        for key, ft_module in base_param_dict.items():
            set_module(pipe.transformer, key, ft_module)
            
        with torch.no_grad():
            noise_pred_null = pipe.transformer(
                hidden_states=xt,
                timestep=timesteps / 1000,
                guidance=guidance,
                pooled_projections=scene_data['null_pooled_prompt_embeds'],
                encoder_hidden_states=scene_data['null_prompt_embeds'],
                txt_ids=scene_data['text_ids'],
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]
            
            noise_pred_from = pipe.transformer(
                hidden_states=xt,
                timestep=timesteps / 1000,
                guidance=guidance,
                pooled_projections=scene_data['erase_from_pooled_prompt_embeds'],
                encoder_hidden_states=scene_data['erase_from_prompt_embeds'],
                txt_ids=scene_data['text_ids'],
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]
            
            noise_pred_erase = pipe.transformer(
                hidden_states=xt,
                timestep=timesteps / 1000,
                guidance=guidance,
                pooled_projections=scene_data['erase_pooled_prompt_embeds'],
                encoder_hidden_states=scene_data['erase_prompt_embeds'],
                txt_ids=scene_data['text_ids'],
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]
        
        
        for key, ft_module in esd_param_dict.items():
            set_module(pipe.transformer, key, ft_module)
        pipe.transformer.train()
        
        
        model_pred = pipe.transformer(
            hidden_states=xt,
            timestep=timesteps / 1000,
            guidance=guidance,
            pooled_projections=scene_data['erase_from_pooled_prompt_embeds'],
            encoder_hidden_states=scene_data['erase_from_prompt_embeds'],
            txt_ids=scene_data['text_ids'],
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]
        
        
        negative_guidance_value = scene_data['negative_guidance']
        target = noise_pred_from - negative_guidance_value * (noise_pred_erase - noise_pred_null)
        loss = torch.mean(((model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1), 1,)
        loss = loss.mean()
        
        
        loss.backward()
        
        grad_norm = esd_params[-1].grad
        grad_norm = grad_norm.norm().item() if grad_norm is not None else -100.0
        
        optimizer.step()
        optimizer.zero_grad()
        pipe.transformer.zero_grad()
        
        global_step += 1
        pbar.update()
        
        
        scene_losses[scene_name].append(loss.item())
        loss_history['esd_loss'] = loss_history.get('esd_loss', []) + [loss.item()]
        
        
        pbar.set_postfix({
            'scene': scene_name,
            'scene_iter': f"{scene_counters[scene_name]}/{iterations_per_scene}",
            'grad_norm': f'{grad_norm:.4f}',
            'loss': f'{loss.item():.4f}',
            'neg_guid': f'{negative_guidance_value:.1f}',
        })
        
        
        if global_step % save_interval == 0 and global_step > 0:
            checkpoint_path = f"{save_path}/checkpoint_{global_step}.safetensors"
            save_file(esd_param_dict, checkpoint_path)
            print(f"\nSaved checkpoint: {checkpoint_path}")
            
            
            print("Scene training progress:")
            for scene in scenes:
                name = scene['erase_from']
                count = scene_counters[name]
                avg_loss = sum(scene_losses[name][-100:]) / len(scene_losses[name][-100:]) if scene_losses[name] else 0
                print(f"  {name}: {count}/{iterations_per_scene} iterations, avg_loss={avg_loss:.4f}")
        
        
        model_pred = loss = target = xt = noise_pred_null = noise_pred_from = noise_pred_erase = None
        torch.cuda.empty_cache()
        gc.collect()

    
    if len(scenes) == 1:
        
        scene_name = scenes[0]['erase_from']
        final_save_path = f"{save_path}/esd-{erase_concept.replace(' ', '_')}-from-{scene_name.replace(' ', '_')}-{train_method.replace('-','')}.safetensors"
    else:
        
        final_save_path = f"{save_path}/esd-{erase_concept.replace(' ', '_')}-multi_{len(scenes)}scenes-{train_method.replace('-','')}.safetensors"
    
    save_file(esd_param_dict, final_save_path)
    print(f"\nTraining completed! Final model saved to: {final_save_path}")
    
    
    print("\nFinal training statistics:")
    for scene in scenes:
        name = scene['erase_from']
        count = scene_counters[name]
        avg_loss = sum(scene_losses[name]) / len(scene_losses[name]) if scene_losses[name] else 0
        print(f"  {name}: {count} iterations, final_avg_loss={avg_loss:.4f}")
    
    pipe.transformer.eval()
    pipe = pipe.to(device)
    torch.set_grad_enabled(False)