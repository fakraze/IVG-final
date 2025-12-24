"""
Calibration script for CacheQuant + Tune-A-Video
This script collects calibration data to determine optimal cache schedule (DPS)
and error compensation parameters (DEC).
"""

import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
import logging

from tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from tuneavideo.models.unet import UNet3DConditionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureCollector:
    """Collect intermediate features from UNet during denoising."""
    
    def __init__(self):
        self.features = []
        self.hooks = []
    
    def hook_fn(self, module, input, output):
        """Hook function to capture layer outputs - only keep spatial mean to save memory."""
        # Reduce feature to spatial mean to save massive memory
        # From [B, C, F, H, W] -> [B, C, F] -> mean -> scalar
        feature_mean = output.detach().mean(dim=[3, 4]).cpu()  # Keep only [B, C, F]
        self.features.append(feature_mean)
    
    def register_hooks(self, model, layer_name='down_blocks'):
        """Register hooks to collect features from down_blocks."""
        hook_registered = False
        for name, module in model.named_modules():
            # Hook one of the down_blocks to collect timestep features
            if 'down_blocks.2' in name and 'resnets.1' in name and not hook_registered:
                hook = module.register_forward_hook(self.hook_fn)
                self.hooks.append(hook)
                logger.info(f"Registered hook at: {name}")
                hook_registered = True
                break
        
        if not hook_registered:
            logger.warning("No hook was registered! Features will not be collected.")
        
        return hook_registered
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def reset(self):
        """Reset collected features."""
        self.features = []
        torch.cuda.empty_cache()


def compute_feature_similarity(features):
    """
    Compute similarity between consecutive features to determine cache schedule.
    High similarity means we can cache more aggressively.
    """
    similarities = []
    for i in range(len(features) - 1):
        f1 = features[i].flatten().float()  # Convert to float32
        f2 = features[i + 1].flatten().float()  # Convert to float32
        # Compute cosine similarity
        sim = torch.nn.functional.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0))
        similarities.append(sim.item())
    return similarities


def compute_optimal_cache_schedule(similarities, cache_interval=5, num_steps=50):
    """
    Compute optimal cache schedule based on feature similarities using DPS.
    
    DPS Strategy:
    - Start with uniform intervals as baseline
    - Adjust based on similarity: skip high-similarity regions, compute more at low-similarity regions
    - Maintain interval constraint: cache_interval/2 <= interval <= 2*cache_interval
    """
    # Convert to numpy for easier manipulation
    sims = np.array(similarities)
    
    # Normalize similarities to [0, 1]
    sims_norm = (sims - sims.min()) / (sims.max() - sims.min() + 1e-8)
    
    # DPS: Use similarity to determine adaptive intervals
    # High similarity (close to 1) -> can use longer intervals
    # Low similarity (close to 0) -> need shorter intervals
    
    interval_seq = [0]  # Always start at 0
    current_step = 0
    min_interval = max(1, cache_interval // 2)
    max_interval = cache_interval * 2
    
    while current_step < num_steps:
        # Look ahead to determine next cache point
        if current_step >= len(sims_norm):
            # Near the end, just add the final step
            break
        
        # Compute average similarity in the look-ahead window
        window_end = min(current_step + max_interval, len(sims_norm))
        window_sims = sims_norm[current_step:window_end]
        
        if len(window_sims) == 0:
            break
        
        # High similarity -> use longer interval (can skip more)
        # Low similarity -> use shorter interval (need more computation)
        avg_sim = np.mean(window_sims)
        
        # Adaptive interval: interpolate between min and max based on similarity
        # High similarity (avg_sim close to 1) -> use max_interval
        # Low similarity (avg_sim close to 0) -> use min_interval
        adaptive_interval = int(min_interval + (max_interval - min_interval) * avg_sim)
        adaptive_interval = np.clip(adaptive_interval, min_interval, max_interval)
        
        next_step = current_step + adaptive_interval
        
        if next_step >= num_steps:
            break
        
        interval_seq.append(next_step)
        current_step = next_step
    
    # Always include the last step
    if num_steps not in interval_seq:
        interval_seq.append(num_steps)
    
    return sorted(list(set(interval_seq)))


def calibrate_cachequant(
    model_path,
    pretrained_path,
    inv_latent_path,
    prompt,
    cache_interval=5,
    num_inference_steps=50,
    video_length=24,
    height=512,
    width=512,
    guidance_scale=12.5,
    output_dir="./calibration",
):
    """
    Run calibration to determine optimal cache schedule.
    
    Args:
        model_path: Path to tuned model
        pretrained_path: Path to pretrained stable diffusion
        inv_latent_path: Path to DDIM inverted latent
        prompt: Text prompt for generation
        cache_interval: Target cache interval
        num_inference_steps: Number of denoising steps
        video_length: Number of frames
        height, width: Video dimensions
        guidance_scale: Classifier-free guidance scale
        output_dir: Directory to save calibration data
    """
    
    logger.info("="*60)
    logger.info("Starting CacheQuant Calibration for Tune-A-Video")
    logger.info("="*60)
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    
    # Clear cache before loading
    torch.cuda.empty_cache()
    
    device = torch.device("cuda")
    
    unet = UNet3DConditionModel.from_pretrained(
        model_path, 
        subfolder='unet', 
        torch_dtype=torch.float16
    ).to('cuda')
    
    pipe = TuneAVideoPipeline.from_pretrained(
        pretrained_path, 
        unet=unet, 
        torch_dtype=torch.float16
    ).to("cuda")
    
    # Enable memory efficient attention if available
    try:
        pipe.enable_xformers_memory_efficient_attention()
        logger.info("Enabled xformers memory efficient attention")
    except:
        logger.warning("xformers not available, using standard attention")
    
    # Load inverted latent
    logger.info(f"Loading inverted latent from {inv_latent_path}")
    ddim_inv_latent = torch.load(inv_latent_path).to(torch.float16)
    
    # Clear cache
    torch.cuda.empty_cache()
    
    # Setup feature collector
    collector = FeatureCollector()
    collector.register_hooks(pipe.unet)
    
    # Run inference to collect features
    logger.info("Running inference to collect features...")
    logger.info("(Using reduced feature representation to save memory)")
    logger.info("(Skipping VAE decode to save memory)")
    with torch.no_grad(), torch.cuda.amp.autocast():
        # Get text embeddings first
        text_embeddings = pipe._encode_prompt(
            prompt,
            device,
            num_videos_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=""
        )
        
        # We only need to run the denoising loop, not decode
        # So we'll manually call the denoising process
        latents = pipe.prepare_latents(
            1,  # batch_size
            pipe.unet.in_channels,
            video_length,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator=None,
            latents=ddim_inv_latent
        )
        
        # Setup scheduler
        pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = pipe.scheduler.timesteps
        
        # Denoising loop - only run UNet, skip VAE decode
        from tqdm import tqdm
        for i, t in enumerate(tqdm(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if True else latents
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Don't decode - we only needed the features during denoising
        logger.info("Denoising complete (skipped VAE decode)")
    
    # Remove hooks and clear cache
    collector.remove_hooks()
    torch.cuda.empty_cache()
    
    logger.info(f"Collected {len(collector.features)} feature maps")
    
    # Compute feature similarities
    logger.info("Computing feature similarities...")
    similarities = compute_feature_similarity(collector.features)
    
    # Clear features to free memory
    collector.features = []
    torch.cuda.empty_cache()
    
    # Compute optimal cache schedule
    logger.info("Computing optimal cache schedule (DPS)...")
    interval_seq = compute_optimal_cache_schedule(
        similarities, 
        cache_interval=cache_interval,
        num_steps=num_inference_steps
    )
    
    logger.info(f"Optimal cache schedule: {interval_seq}")
    logger.info(f"Number of full UNet computations: {len(interval_seq)}/{num_inference_steps}")
    logger.info(f"Theoretical speedup: {num_inference_steps/len(interval_seq):.2f}x")
    
    # Save calibration data
    os.makedirs(output_dir, exist_ok=True)
    calibration_data = {
        'interval_seq': interval_seq,
        'cache_interval': cache_interval,
        'num_steps': num_inference_steps,
        'similarities': similarities,
        'model_path': model_path,
        'prompt': prompt,
    }
    
    output_path = os.path.join(
        output_dir, 
        f"cachequant_cache{cache_interval}_steps{num_inference_steps}.pth"
    )
    torch.save(calibration_data, output_path)
    logger.info(f"Saved calibration data to {output_path}")
    
    logger.info("="*60)
    logger.info("Calibration completed successfully!")
    logger.info("="*60)
    
    return calibration_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate CacheQuant for Tune-A-Video")
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to tuned Tune-A-Video model (e.g., ./outputs/car-turn)"
    )
    parser.add_argument(
        "--pretrained-path",
        type=str,
        default="./checkpoints/stable-diffusion-v1-4",
        help="Path to pretrained Stable Diffusion model"
    )
    parser.add_argument(
        "--inv-latent",
        type=str,
        required=True,
        help="Path to DDIM inverted latent (e.g., ./outputs/car-turn/inv_latents/ddim_latent-100.pt)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for generation"
    )
    parser.add_argument(
        "--cache-interval",
        type=int,
        default=5,
        help="Target cache interval (default: 5)"
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of denoising steps (default: 50)"
    )
    parser.add_argument(
        "--video-length",
        type=int,
        default=24,
        help="Number of frames (default: 24)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Video height (default: 512)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Video width (default: 512)"
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=12.5,
        help="Classifier-free guidance scale (default: 12.5)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./calibration",
        help="Directory to save calibration data (default: ./calibration)"
    )
    
    args = parser.parse_args()
    
    calibrate_cachequant(
        model_path=args.model_path,
        pretrained_path=args.pretrained_path,
        inv_latent_path=args.inv_latent,
        prompt=args.prompt,
        cache_interval=args.cache_interval,
        num_inference_steps=args.num_inference_steps,
        video_length=args.video_length,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        output_dir=args.output_dir,
    )
