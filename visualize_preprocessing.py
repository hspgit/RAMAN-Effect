import os
import numpy as np
import ramanspy as rp
import matplotlib.pyplot as plt

def main():
    # Setup output directory
    out_dir = "preprocessing_visualizations"
    os.makedirs(out_dir, exist_ok=True)
    
    # Load test data and wavenumbers
    print("Loading data...")
    X_path = "data/X_test.npy"
    wavenumbers_path = "data/wavenumbers.npy"
    
    X = np.load(X_path).astype(np.float32)
    wavenumbers = np.load(wavenumbers_path)
    
    # Select examples for visualization
    sample_indices = [0, 50, 100]
    examples = [rp.Spectrum(X[idx], wavenumbers) for idx in sample_indices]
    
    # Define preprocessing steps based on src/dataset.py
    steps = [
        ("1_Despiking", "Despiking (Whitaker-Hayes)", rp.preprocessing.despike.WhitakerHayes(), "Removes cosmic ray spikes from the spectrum."),
        ("2_Denoising", "Denoising (Savitzky-Golay)", rp.preprocessing.denoise.SavGol(window_length=9, polyorder=3), "Smoothes the signal to reduce high-frequency noise."),
        ("3_Baseline_Correction", "Baseline Correction (ASPLS)", rp.preprocessing.baseline.ASPLS(), "Removes the fluorescent baseline background. Notice how it behaves on this dataset!"),
        ("4_Normalization", "Normalization (MinMax)", rp.preprocessing.normalise.MinMax(), "Scales the intensities to a fixed range (typically 0 to 1).")
    ]
    
    image_files = []
    
    # We will plot the raw data first
    plt.figure(figsize=(15, 4))
    for i, ex in enumerate(examples):
        plt.subplot(1, 3, i+1)
        plt.plot(wavenumbers, ex.spectral_data, color='blue', linewidth=1)
        plt.title(f"Sample {sample_indices[i]} - Raw")
        plt.xlabel("Wavenumber (cm⁻¹)")
        plt.ylabel("Intensity")
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    raw_file = "0_Raw_Data.png"
    plt.savefig(os.path.join(out_dir, raw_file), dpi=150)
    plt.close()
    
    image_files.append(("Raw Data", raw_file, "The original unprocessed Raman spectra. Note that the data in `X_test.npy` appears to already be min-max normalized (range 0 to 1)."))

    for step_id, step_title, step_func, step_desc in steps:
        print(f"Applying: {step_title}")
        plt.figure(figsize=(15, 4))
        
        for i in range(len(examples)):
            prev_obj = examples[i]
            prev_data = prev_obj.spectral_data.copy()
            
            # Apply step
            new_obj = step_func.apply(prev_obj)
            examples[i] = new_obj
            new_data = new_obj.spectral_data
            
            plt.subplot(1, 3, i+1)
            
            if "Baseline" in step_title:
                baseline = prev_data - new_data
                plt.plot(wavenumbers, prev_data, label='Before', color='lightgray', linewidth=2)
                plt.plot(wavenumbers, baseline, label='Estimated Baseline', color='green', linewidth=1, linestyle='--')
                plt.plot(wavenumbers, new_data, label='After', color='red', linewidth=1, alpha=0.8)
            else:
                plt.plot(wavenumbers, prev_data, label='Before', color='lightgray', linewidth=2)
                plt.plot(wavenumbers, new_data, label='After', color='red', linewidth=1, alpha=0.8)
                
            plt.title(f"Sample {sample_indices[i]} - {step_title}")
            plt.xlabel("Wavenumber (cm⁻¹)")
            if i == 0:
                plt.ylabel("Intensity")
            plt.legend(fontsize=8)
            plt.grid(True, alpha=0.3)
            
        plt.tight_layout()
        filename = f"{step_id}.png"
        filepath = os.path.join(out_dir, filename)
        plt.savefig(filepath, dpi=150)
        plt.close()
        
        image_files.append((step_title, filename, step_desc))
        
    # Generate Markdown file
    md_path = os.path.join(out_dir, "preprocessing_steps.md")
    with open(md_path, "w") as f:
        f.write("# Raman Spectra Preprocessing Pipeline\n\n")
        f.write("This document visualizes the effect of each preprocessing step on three example Raman spectra from the dataset (`X_test.npy`).\n")
        f.write("By overlaying the 'Before' and 'After' states, we can see exactly what each transformation does to the data. This visual analysis helps explain why the pipeline might be reducing classification accuracy compared to raw data (e.g., ASPLS struggling with already-normalized data).\n\n")
        
        for step_title, filename, step_desc in image_files:
            f.write(f"## {step_title}\n\n")
            f.write(f"{step_desc}\n\n")
            f.write(f"![{step_title}]({filename})\n\n")
            
    print(f"Visualizations and markdown successfully saved to {out_dir}/")

if __name__ == "__main__":
    main()
