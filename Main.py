# -*- coding: utf-8 -*-
"""
Complete ASR Model Similarity Evaluation - GPU Enhanced
Enhanced version with GPU support, visualization and additional analysis
"""
import os

# Make specific GPUs visible
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor, Wav2Vec2Config, HubertModel, HubertConfig, AutoFeatureExtractor
import soundfile as sf
import torch
import torch.nn.functional as F
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import pandas as pd
import gc

def get_device():
    """Get the best available device (CUDA > MPS > CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def extract_reps(model_name, model_class, feat_extractor, model_config, device, batch_size=4):
    """Extract hidden state representations from speech models with GPU support"""
    print(f"Loading model: {model_name}")
    
    # Initialize model configuration
    configuration = model_config.from_pretrained(model_name, output_hidden_states=True)
    model = model_class(configuration)
    model.eval()  # Set to evaluation mode
    model = model.to(device)  # Move model to GPU
    
    # Initialize feature extractor
    feature_extractor = feat_extractor.from_pretrained(model_name)
    
    # List to store hidden state representations
    all_hidden_states = []
    data_dir = "Speech_folder/"
    
    if not os.path.exists(data_dir):
        print(f"Warning: {data_dir} does not exist. Please create this directory and add your audio files.")
        return []
    
    audio_files = []
    # Collect all audio files
    for root, _, files in os.walk(data_dir):
        for filename in files:
            if filename.endswith((".flac", ".wav", ".mp3")):
                audio_files.append(os.path.join(root, filename))
    
    print(f"Found {len(audio_files)} audio files")
    
    # Process files in batches to manage GPU memory
    for i in range(0, len(audio_files), batch_size):
        batch_files = audio_files[i:i+batch_size]
        batch_hidden_states = []
        
        if i % (batch_size * 5) == 0:
            print(f"Processing batch {i//batch_size + 1}/{(len(audio_files) + batch_size - 1)//batch_size}")
            clear_gpu_memory()  # Clear memory periodically
        
        for audio_path in batch_files:
            try:
                # Read audio file
                audio, sample_rate = sf.read(audio_path)
                
                # Resample if necessary (you might want to add actual resampling here)
                if sample_rate != 16000:
                    print(f"Warning: {audio_path} has sample rate {sample_rate}, expected 16000")
                
                # Limit audio length to prevent memory issues (max 30 seconds)
                max_length = 16000 * 30
                if len(audio) > max_length:
                    audio = audio[:max_length]
                
                # Extract features
                inputs = feature_extractor(audio, return_tensors="pt", sampling_rate=16000, padding=True)
                
                # Move inputs to GPU
                input_values = inputs.input_values.to(device)
                
                # Extract hidden states
                with torch.no_grad():
                    outputs = model(input_values)
                
                # Move hidden states to CPU to save GPU memory
                hidden_states = []
                for hidden_state in outputs.hidden_states:
                    hidden_states.append(hidden_state.cpu())
                
                batch_hidden_states.append(hidden_states)
                
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                continue
        
        all_hidden_states.extend(batch_hidden_states)
        
        # Clear GPU memory after each batch
        if device.type == "cuda":
            torch.cuda.empty_cache()
    
    print(f"Successfully processed {len(all_hidden_states)} files")
    
    # Clear model from GPU memory
    del model
    clear_gpu_memory()
    
    return all_hidden_states

# CKA Similarity Functions (GPU-optimized)
def centering_gpu(K, device):
    """Center the kernel matrix on GPU"""
    n = K.shape[0]
    unit = torch.ones([n, n], device=device)
    I = torch.eye(n, device=device)
    H = I - unit / n
    return torch.mm(torch.mm(H, K), H)

def rbf_gpu(X, sigma=None, device=None):
    """RBF kernel on GPU"""
    if device is None:
        device = X.device
    
    GX = torch.mm(X, X.T)
    KX = torch.diag(GX).unsqueeze(1) - GX + (torch.diag(GX).unsqueeze(1) - GX).T
    
    if sigma is None:
        mdist = torch.median(KX[KX != 0])
        sigma = torch.sqrt(mdist)
    
    KX = KX * (-0.5 / (sigma * sigma))
    KX = torch.exp(KX)
    return KX

def kernel_HSIC_gpu(X, Y, sigma, device):
    """Kernel HSIC on GPU"""
    K_X = rbf_gpu(X, sigma, device)
    K_Y = rbf_gpu(Y, sigma, device)
    return torch.sum(centering_gpu(K_X, device) * centering_gpu(K_Y, device))

def linear_HSIC_gpu(X, Y, device):
    """Linear HSIC on GPU"""
    L_X = torch.mm(X, X.T)
    L_Y = torch.mm(Y, Y.T)
    return torch.sum(centering_gpu(L_X, device) * centering_gpu(L_Y, device))

def linear_CKA_gpu(X, Y, device):
    """Linear CKA similarity on GPU"""
    hsic = linear_HSIC_gpu(X, Y, device)
    var1 = torch.sqrt(linear_HSIC_gpu(X, X, device))
    var2 = torch.sqrt(linear_HSIC_gpu(Y, Y, device))
    return hsic / (var1 * var2)

def kernel_CKA_gpu(X, Y, sigma=None, device=None):
    """Kernel CKA similarity on GPU"""
    if device is None:
        device = X.device
    
    hsic = kernel_HSIC_gpu(X, Y, sigma, device)
    var1 = torch.sqrt(kernel_HSIC_gpu(X, X, sigma, device))
    var2 = torch.sqrt(kernel_HSIC_gpu(Y, Y, sigma, device))
    return hsic / (var1 * var2)

def calculate_cka_similarity_gpu(rep1, rep2, device, use_linear=True, max_samples=1000):
    """Calculate CKA similarity matrix between two models on GPU"""
    num_layers1 = len(rep1) - 1  # Exclude input embeddings
    num_layers2 = len(rep2) - 1
    
    cka_matrix = torch.zeros((num_layers1, num_layers2), device='cpu')
    
    for i in range(num_layers1):
        for j in range(num_layers2):
            # Get representations and move to GPU
            reps1 = rep1[i+1].squeeze(0)  # Skip input layer
            reps2 = rep2[j+1].squeeze(0)
            
            # Subsample if too many samples (to manage memory)
            if reps1.shape[0] > max_samples:
                indices = torch.randperm(reps1.shape[0])[:max_samples]
                reps1 = reps1[indices]
                reps2 = reps2[indices]
            
            # Move to GPU
            reps1 = reps1.to(device)
            reps2 = reps2.to(device)
            
            # Calculate CKA
            if use_linear:
                cka_value = linear_CKA_gpu(reps1, reps2, device)
            else:
                cka_value = kernel_CKA_gpu(reps1, reps2, device=device)
            
            # Move result back to CPU
            cka_matrix[i, j] = cka_value.cpu()
            
            # Clear GPU memory
            if device.type == "cuda":
                torch.cuda.empty_cache()
    
    return cka_matrix.numpy()

def average_cka_similarity_gpu(hs1_list, hs2_list, device, use_linear=True, max_samples=1000):
    """Calculate average CKA similarity across multiple audio files on GPU"""
    if len(hs1_list) == 0 or len(hs2_list) == 0:
        print("Error: No hidden states to compare")
        return None
        
    n_audio = min(len(hs1_list), len(hs2_list))
    
    if n_audio == 0:
        return None
    
    # Calculate for first file to get dimensions
    first_cka = calculate_cka_similarity_gpu(hs1_list[0], hs2_list[0], device, use_linear, max_samples)
    sum_mat = first_cka.copy()
    
    # Average over all audio files
    for i in range(1, n_audio):
        temp_mat = calculate_cka_similarity_gpu(hs1_list[i], hs2_list[i], device, use_linear, max_samples)
        sum_mat += temp_mat
        
        if i % 10 == 0:
            print(f"Processed {i+1}/{n_audio} audio files")
            clear_gpu_memory()
    
    return sum_mat / n_audio

def plot_heatmap(cka_matrix, title, xlabel="Model 2 Layers", ylabel="Model 1 Layers", save_path=None):
    """Plot CKA similarity heatmap"""
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(cka_matrix, 
                annot=True, 
                fmt='.3f', 
                cmap='viridis', 
                cbar_kws={'label': 'CKA Similarity'},
                xticklabels=range(1, cka_matrix.shape[1] + 1),
                yticklabels=range(1, cka_matrix.shape[0] + 1))
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to {save_path}")
    
    plt.show()
    
    # Print statistics
    print(f"\n{title} Statistics:")
    print(f"Mean CKA: {np.mean(cka_matrix):.3f}")
    print(f"Max CKA: {np.max(cka_matrix):.3f}")
    print(f"Min CKA: {np.min(cka_matrix):.3f}")
    print(f"Std CKA: {np.std(cka_matrix):.3f}")

def analyze_diagonal_similarity(cka_matrix, model1_name, model2_name):
    """Analyze diagonal similarities (layer-to-layer correspondence)"""
    min_dim = min(cka_matrix.shape[0], cka_matrix.shape[1])
    diagonal_values = [cka_matrix[i, i] for i in range(min_dim)]
    
    print(f"\nDiagonal Analysis ({model1_name} vs {model2_name}):")
    print(f"Mean diagonal CKA: {np.mean(diagonal_values):.3f}")
    print(f"Layer correspondences (CKA > 0.5): {sum(1 for x in diagonal_values if x > 0.5)}/{min_dim}")
    print(f"Layer correspondences (CKA > 0.7): {sum(1 for x in diagonal_values if x > 0.7)}/{min_dim}")
    
    return diagonal_values

def print_gpu_memory_usage():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        current_memory = torch.cuda.memory_allocated() / 1e9
        max_memory = torch.cuda.max_memory_allocated() / 1e9
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory - Current: {current_memory:.1f} GB, Max used: {max_memory:.1f} GB, Total: {total_memory:.1f} GB")

def main():
    """Main evaluation function with GPU support"""
    print("Starting GPU-Enabled ASR Model Similarity Evaluation...")
    
    # Get device
    device = get_device()
    
    # Model specifications
    models = {
        'wav2vec2_base': ("facebook/wav2vec2-base-960h", Wav2Vec2Model, Wav2Vec2FeatureExtractor, Wav2Vec2Config),
        'wav2vec2_large': ("facebook/wav2vec2-large-960h", Wav2Vec2Model, Wav2Vec2FeatureExtractor, Wav2Vec2Config),
        'xlsr_large': ("jonatasgrosman/wav2vec2-large-xlsr-53-english", Wav2Vec2Model, Wav2Vec2FeatureExtractor, Wav2Vec2Config),
        'hubert_base': ("facebook/hubert-base-ls960", HubertModel, AutoFeatureExtractor, HubertConfig),
        'hubert_large': ("facebook/hubert-large-ll60k", HubertModel, AutoFeatureExtractor, HubertConfig)
    }
    
    # Extract representations for all models
    model_reps = {}
    batch_size = 2 if device.type == "cuda" else 4  # Smaller batches for GPU to manage memory
    
    for model_name, (model_id, model_class, feat_extractor, config) in models.items():
        print(f"\n--- Processing {model_name} ---")
        print_gpu_memory_usage()
        
        model_reps[model_name] = extract_reps(model_id, model_class, feat_extractor, config, device, batch_size)
        
        print_gpu_memory_usage()
        print(f"Extracted representations for {len(model_reps[model_name])} audio files")
    
    # Calculate similarities (using HuBERT-Large as reference)
    reference_model = 'hubert_large'
    comparisons = [
        ('wav2vec2_base', 'Wav2Vec2 Base vs HuBERT Large'),
        ('wav2vec2_large', 'Wav2Vec2 Large vs HuBERT Large'), 
        ('xlsr_large', 'XLSR Large vs HuBERT Large'),
        ('hubert_base', 'HuBERT Base vs HuBERT Large')
    ]
    
    results = {}
    
    for model_name, title in comparisons:
        print(f"\n--- Computing CKA similarity: {title} ---")
        print_gpu_memory_usage()
        
        if len(model_reps[model_name]) > 0 and len(model_reps[reference_model]) > 0:
            if len(model_reps[model_name]) > 1:
                # Multiple audio files - use average
                cka_matrix = average_cka_similarity_gpu(
                    model_reps[model_name], 
                    model_reps[reference_model], 
                    device,
                    use_linear=True,  # Linear CKA is faster and uses less memory
                    max_samples=500   # Limit samples per layer to manage memory
                )
            else:
                # Single audio file
                cka_matrix = calculate_cka_similarity_gpu(
                    model_reps[model_name][0], 
                    model_reps[reference_model][0], 
                    device,
                    use_linear=True,
                    max_samples=500
                )
            
            if cka_matrix is not None:
                results[model_name] = cka_matrix
                
                # Plot heatmap
                plot_heatmap(cka_matrix, 
                            title, 
                            xlabel="HuBERT Large Layers",
                            ylabel=f"{model_name.replace('_', ' ').title()} Layers",
                            save_path=f"{model_name}_cka_heatmap_gpu.png")
                
                # Analyze diagonal
                analyze_diagonal_similarity(cka_matrix, model_name, reference_model)
                
                print_gpu_memory_usage()
        else:
            print(f"Skipping {model_name} - no representations available")
    
    # Final memory cleanup
    clear_gpu_memory()
    print("\nEvaluation completed!")
    
    return results

if __name__ == "__main__":
    results = main()
