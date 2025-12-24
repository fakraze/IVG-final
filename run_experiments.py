"""
Comprehensive experiments for Tune-A-Video + CacheQuant
Evaluates:
1. Frame Consistency (CLIP score between frames)
2. Textual Alignment (CLIP score with prompt)
3. Inference time
4. Different cache intervals (3, 5, 10)
5. With/without calibration (DPS)
"""

import os
import sys
import yaml
import torch
import time
import csv
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image
import clip

from tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from tuneavideo.pipelines.pipeline_tuneavideo_cachequant import TuneAVideoCacheQuantPipeline
from tuneavideo.models.unet import UNet3DConditionModel
from tuneavideo.util import save_videos_grid


class ExperimentRunner:
    def __init__(self, output_dir="./experiment_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        self.videos_dir = self.output_dir / "videos"
        self.videos_dir.mkdir(exist_ok=True)
        self.calibration_dir = self.output_dir / "calibration"
        self.calibration_dir.mkdir(exist_ok=True)
        
        # Load CLIP model for evaluation
        print("Loading CLIP model for evaluation...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device="cuda")
        
        # Experiment configurations
        self.configs = [
            "configs/car-turn.yaml",
            "configs/man-skiing.yaml",
            "configs/man-surfing.yaml",
            "configs/rabbit-watermelon.yaml"
        ]
        
        self.cache_intervals = [3, 5, 10]
        
        # Results storage
        self.results = []
        self.calibration_times = {}  # Track calibration time per model
        
    def load_config(self, config_path):
        """Load experiment configuration from yaml."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def compute_frame_consistency(self, video_frames):
        """
        Compute frame consistency using CLIP.
        Returns average cosine similarity between all pairs of frames.
        """
        # video_frames: numpy array [num_frames, H, W, C]
        features = []
        
        for frame in video_frames:
            # Convert to PIL Image
            frame_pil = Image.fromarray((frame * 255).astype(np.uint8))
            # Preprocess and encode
            frame_input = self.clip_preprocess(frame_pil).unsqueeze(0).to("cuda")
            with torch.no_grad():
                feature = self.clip_model.encode_image(frame_input)
                feature = feature / feature.norm(dim=-1, keepdim=True)
            features.append(feature)
        
        # Compute pairwise cosine similarities
        features = torch.cat(features, dim=0)
        similarities = []
        num_frames = features.shape[0]
        
        for i in range(num_frames):
            for j in range(i + 1, num_frames):
                sim = torch.cosine_similarity(features[i:i+1], features[j:j+1]).item()
                similarities.append(sim)
        
        # Return average similarity
        return np.mean(similarities) * 100  # Scale to match paper format
    
    def compute_textual_alignment(self, video_frames, prompt):
        """
        Compute textual alignment using CLIP.
        Returns average CLIP score between frames and prompt.
        """
        # Encode text
        text_input = clip.tokenize([prompt]).to("cuda")
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_input)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Encode frames and compute similarities
        similarities = []
        for frame in video_frames:
            frame_pil = Image.fromarray((frame * 255).astype(np.uint8))
            frame_input = self.clip_preprocess(frame_pil).unsqueeze(0).to("cuda")
            with torch.no_grad():
                image_features = self.clip_model.encode_image(frame_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                sim = torch.cosine_similarity(image_features, text_features).item()
                similarities.append(sim)
        
        return np.mean(similarities) * 100  # Scale to match paper format
    
    def run_calibration(self, config, model_path, inv_latent_path, prompt):
        """Run calibration for DPS optimization."""
        from calibrate_tuneavideo_cachequant import calibrate_cachequant
        
        model_name = Path(model_path).name
        calib_output = self.calibration_dir / f"{model_name}_calib.pth"
        
        if calib_output.exists():
            print(f"  Calibration already exists: {calib_output}")
            return str(calib_output)
        
        print(f"  Running calibration for {model_name}...")
        try:
            # Clear GPU memory before calibration
            torch.cuda.empty_cache()
            
            # Measure calibration time
            import time
            calib_start = time.time()
            
            calibrate_cachequant(
                model_path=model_path,
                pretrained_path=config['pretrained_model_path'],
                inv_latent_path=inv_latent_path,
                prompt=prompt,
                cache_interval=5,  # Default
                num_inference_steps=config['validation_data']['num_inference_steps'],
                video_length=config['validation_data']['video_length'],
                height=config['validation_data']['height'],
                width=config['validation_data']['width'],
                guidance_scale=config['validation_data']['guidance_scale'],
                output_dir=str(self.calibration_dir)
            )
            # Rename to model-specific name
            default_calib = self.calibration_dir / "cachequant_cache5_steps50.pth"
            if default_calib.exists():
                default_calib.rename(calib_output)
            
            # Record calibration time
            calib_time = time.time() - calib_start
            self.calibration_times[model_name] = calib_time
            print(f"  Calibration completed in {calib_time:.2f}s")
            
            # Clear GPU memory after calibration
            torch.cuda.empty_cache()
            
            return str(calib_output)
        except Exception as e:
            print(f"  Calibration failed: {e}")
            print(f"  Will continue experiments without DPS for this model")
            return None
    
    def run_inference(self, config, model_path, inv_latent_path, prompt, 
                     use_cache=False, cache_interval=5, calibration_path=None,
                     output_name="output"):
        """
        Run inference and return video, inference time, and metrics.
        """
        print(f"    Running inference: {output_name}")
        
        # Clear GPU memory before inference
        torch.cuda.empty_cache()
        
        # Load model
        unet = UNet3DConditionModel.from_pretrained(
            model_path, 
            subfolder='unet', 
            torch_dtype=torch.float16
        ).to('cuda')
        
        if use_cache:
            pipe = TuneAVideoCacheQuantPipeline.from_pretrained(
                config['pretrained_model_path'], 
                unet=unet, 
                torch_dtype=torch.float16
            ).to("cuda")
            
            if calibration_path:
                pipe.load_cachequant_params(
                    cache_interval=cache_interval,
                    calibration_path=calibration_path,
                    use_dec=False
                )
            else:
                pipe.load_cachequant_params(
                    cache_interval=cache_interval,
                    calibration_path=None,
                    use_dec=False
                )
        else:
            pipe = TuneAVideoPipeline.from_pretrained(
                config['pretrained_model_path'], 
                unet=unet, 
                torch_dtype=torch.float16
            ).to("cuda")
        
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except:
            pass
        pipe.enable_vae_slicing()
        
        # Load inverted latent
        ddim_inv_latent = torch.load(inv_latent_path).to(torch.float16)
        
        # Run inference and measure time
        torch.cuda.synchronize()
        start_time = time.time()
        
        video = pipe(
            prompt,
            latents=ddim_inv_latent,
            video_length=config['validation_data']['video_length'],
            height=config['validation_data']['height'],
            width=config['validation_data']['width'],
            num_inference_steps=config['validation_data']['num_inference_steps'],
            guidance_scale=config['validation_data']['guidance_scale']
        ).videos
        
        torch.cuda.synchronize()
        inference_time = time.time() - start_time
        
        # Save video
        video_path = self.videos_dir / f"{output_name}.gif"
        save_videos_grid(video, str(video_path))
        
        # Convert to numpy for evaluation
        if isinstance(video, torch.Tensor):
            video_np = video.cpu().numpy()
        else:
            video_np = video
        
        # video_np shape: [batch, channels, frames, height, width]
        # Convert to [frames, height, width, channels]
        video_np = video_np[0].transpose(1, 2, 3, 0)  # [frames, H, W, C]
        
        # Compute metrics
        frame_consistency = self.compute_frame_consistency(video_np)
        textual_alignment = self.compute_textual_alignment(video_np, prompt)
        
        # Clean up
        del pipe, unet, video, video_np, ddim_inv_latent
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        
        return {
            'video_path': str(video_path),
            'inference_time': inference_time,
            'frame_consistency': frame_consistency,
            'textual_alignment': textual_alignment
        }
    
    def run_experiments(self):
        """Run all experiments."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("="*80)
        print("Starting Comprehensive Experiments")
        print("="*80)
        
        for config_path in self.configs:
            print(f"\n{'='*80}")
            print(f"Config: {config_path}")
            print(f"{'='*80}")
            
            config = self.load_config(config_path)
            model_path = config['output_dir']
            model_name = Path(model_path).name
            
            # Use checkpoint-100 inverted latent
            inv_latent_path = f"{model_path}/inv_latents/ddim_latent-100.pt"
            
            if not os.path.exists(inv_latent_path):
                print(f"  Skipping {model_name}: inverted latent not found")
                continue
            
            # Get prompts
            train_prompt = config['train_data']['prompt']
            val_prompts = config['validation_data']['prompts']
            
            # Run calibration once per model (using train prompt)
            print(f"\n[1/4] Running calibration for DPS...")
            calibration_path = self.run_calibration(
                config, model_path, inv_latent_path, train_prompt
            )
            
            # Experiment 1: Baseline (no cache)
            print(f"\n[2/4] Baseline experiments (no cache)...")
            for i, prompt in enumerate(val_prompts):
                output_name = f"{model_name}_baseline_prompt{i+1}"
                result = self.run_inference(
                    config, model_path, inv_latent_path, prompt,
                    use_cache=False,
                    output_name=output_name
                )
                
                self.results.append({
                    'model': model_name,
                    'prompt_idx': i + 1,
                    'prompt': prompt,
                    'method': 'baseline',
                    'cache_interval': 'N/A',
                    'use_calibration': False,
                    'inference_time': result['inference_time'],
                    'frame_consistency': result['frame_consistency'],
                    'textual_alignment': result['textual_alignment'],
                    'video_path': result['video_path']
                })
            
            # Experiment 2: CacheQuant without calibration
            print(f"\n[3/4] CacheQuant experiments (without calibration)...")
            for cache_interval in self.cache_intervals:
                for i, prompt in enumerate(val_prompts):
                    output_name = f"{model_name}_cache{cache_interval}_nocal_prompt{i+1}"
                    result = self.run_inference(
                        config, model_path, inv_latent_path, prompt,
                        use_cache=True,
                        cache_interval=cache_interval,
                        calibration_path=None,
                        output_name=output_name
                    )
                    
                    self.results.append({
                        'model': model_name,
                        'prompt_idx': i + 1,
                        'prompt': prompt,
                        'method': 'cachequant',
                        'cache_interval': cache_interval,
                        'use_calibration': False,
                        'inference_time': result['inference_time'],
                        'frame_consistency': result['frame_consistency'],
                        'textual_alignment': result['textual_alignment'],
                        'video_path': result['video_path']
                    })
            
            # Experiment 3: CacheQuant with calibration (DPS)
            if calibration_path:
                print(f"\n[4/4] CacheQuant experiments (with calibration/DPS)...")
                for cache_interval in self.cache_intervals:
                    for i, prompt in enumerate(val_prompts):
                        output_name = f"{model_name}_cache{cache_interval}_dps_prompt{i+1}"
                        result = self.run_inference(
                            config, model_path, inv_latent_path, prompt,
                            use_cache=True,
                            cache_interval=cache_interval,
                            calibration_path=calibration_path,
                            output_name=output_name
                        )
                        
                        self.results.append({
                            'model': model_name,
                            'prompt_idx': i + 1,
                            'prompt': prompt,
                            'method': 'cachequant_dps',
                            'cache_interval': cache_interval,
                            'use_calibration': True,
                            'inference_time': result['inference_time'],
                            'frame_consistency': result['frame_consistency'],
                            'textual_alignment': result['textual_alignment'],
                            'video_path': result['video_path']
                        })
            else:
                print(f"\n[4/4] Skipping DPS experiments (calibration failed)")
        
        # Save results
        self.save_results(timestamp)
        self.print_summary()
    
    def save_results(self, timestamp):
        """Save results to CSV files."""
        # Detailed results
        detailed_csv = self.output_dir / f"detailed_results_{timestamp}.csv"
        fieldnames = [
            'model', 'prompt_idx', 'prompt', 'method', 'cache_interval', 
            'use_calibration', 'inference_time', 'frame_consistency', 
            'textual_alignment', 'video_path'
        ]
        
        # Save calibration times separately
        calib_csv = self.output_dir / f"calibration_times_{timestamp}.csv"
        with open(calib_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['model', 'calibration_time_seconds'])
            writer.writeheader()
            for model, calib_time in self.calibration_times.items():
                writer.writerow({'model': model, 'calibration_time_seconds': calib_time})
        print(f"✓ Calibration times saved to: {calib_csv}")
        
        with open(detailed_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)
        
        print(f"\n✓ Detailed results saved to: {detailed_csv}")
        
        # Summary statistics
        summary_csv = self.output_dir / f"summary_results_{timestamp}.csv"
        self.compute_and_save_summary(summary_csv)
    
    def compute_and_save_summary(self, output_path):
        """Compute and save summary statistics."""
        from collections import defaultdict
        import statistics
        
        df = pd.DataFrame(self.results)
        
        # Group by method and cache_interval
        summary = df.groupby(['method', 'cache_interval']).agg({
            'inference_time': ['mean', 'std'],
            'frame_consistency': ['mean', 'std'],
            'textual_alignment': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
        
        # Calculate speedup vs baseline
        baseline_time = df[df['method'] == 'baseline']['inference_time'].mean()
        summary['speedup'] = baseline_time / summary['inference_time_mean']
        
        summary.to_csv(output_path, index=False)
        print(f"✓ Summary results saved to: {output_path}")
    
    def print_summary(self):
        """Print experiment summary."""
        import pandas as pd
        
        df = pd.DataFrame(self.results)
        
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        
        # Group by method
        for method in df['method'].unique():
            method_df = df[df['method'] == method]
            
            print(f"\n{method.upper()}")
            print("-" * 40)
            
            if method == 'baseline':
                print(f"  Avg Time: {method_df['inference_time'].mean():.2f}s")
                print(f"  Avg Frame Consistency: {method_df['frame_consistency'].mean():.2f}")
                print(f"  Avg Textual Alignment: {method_df['textual_alignment'].mean():.2f}")
            else:
                for interval in sorted(method_df['cache_interval'].unique()):
                    interval_df = method_df[method_df['cache_interval'] == interval]
                    baseline_time = df[df['method'] == 'baseline']['inference_time'].mean()
                    speedup = baseline_time / interval_df['inference_time'].mean()
                    
                    print(f"\n  Cache Interval {interval}:")
                    print(f"    Avg Time: {interval_df['inference_time'].mean():.2f}s")
                    print(f"    Speedup: {speedup:.2f}x")
                    print(f"    Avg Frame Consistency: {interval_df['frame_consistency'].mean():.2f}")
                    print(f"    Avg Textual Alignment: {interval_df['textual_alignment'].mean():.2f}")
        
        print("\n" + "="*80)


if __name__ == "__main__":
    runner = ExperimentRunner(output_dir="./experiment_results")
    runner.run_experiments()
