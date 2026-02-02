import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
# import seaborn as sns  # Optional, but helps with defaults if installed, otherwise we manual style

# --- 1. Modern Publication Configuration ---
# Robust Font Selection
try:
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
except Exception:
    mpl.rcParams['font.family'] = 'serif'

# Modern aesthetic tweaks
mpl.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    
    # Lines and Markers
    'lines.linewidth': 2.0,      # Slightly thicker for modern flat look
    'lines.markersize': 7,       # Slightly larger markers
    'lines.markeredgewidth': 1.0,
    
    # Clean Layout
    'axes.grid': True,
    'grid.alpha': 0.2,           # Very faint grid
    'grid.color': '#cccccc',     # Light gray
    'axes.axisbelow': True,      # Grid stays behind data
    'axes.spines.top': False,    # Remove top border (Modern touch)
    'axes.spines.right': False,  # Remove right border (Modern touch)
    
    # PDF Output
    'savefig.format': 'pdf',
    'pdf.fonttype': 42
})

# --- 2. Control Panel (Colors & Labels) ---

key_to_label = {
    'Adam-NSCL': 'Adam-NSCL',
    'Crisp_Hessian': 'CRISPEdit (Hessian)',
    'Crisp_GN_Hessian': 'CRISPEdit (Gauss-Newton)',
    'Crisp_KFAC': 'CRISPEdit (K-FAC)',
    'Crisp_EKFAC': 'CRISPEdit (EK-FAC)',
}

# Pastel Color Scheme
# Note: Modern plots often use slightly saturated pastels for visibility
key_to_color = {
    'Adam-NSCL': '#e5989b',           # Modern Dusty Rose
    'Crisp_Hessian': '#a0af91',      # Kept Intact (Sage)
    'Crisp_GN_Hessian': '#ffcdb2',   # Muted Apricot
    'Crisp_KFAC': '#bde0fe',           # Soft Sky Blue
    'Crisp_EKFAC': '#cdb4db', # Muted Lavender
}

key_to_marker = {
    'Adam-NSCL': 'o', 'Crisp_Hessian': 's', 'Crisp_GN_Hessian': '^',
    'Crisp_KFAC': 'D', 'Crisp_EKFAC': 'v'
}

def plot_metrics(data_list, filename="plots/metrics_comparison.pdf"):
    """
    Takes a dictionary of metric data and plots a comparison.
    """
    try:
        # 4.5 x 3.5 is a good ratio for single column in 2-column paper
        fig, ax = plt.subplots(1, 1, figsize=(5, 3.8))
        
        # --- Plotting Loop ---
        for method, data in data_list.items():
            label = key_to_label.get(method, method)
            color = key_to_color.get(method, '#333333')
            marker = key_to_marker.get(method, 'o')
            
            x_values = np.array(data["pre_accs"])
            y_values = np.array(data['ft_accs'])
            
            # SORTING
            sort_idx = np.argsort(x_values)
            x_sorted = x_values[sort_idx]
            y_sorted = y_values[sort_idx]
            
            ax.plot(x_sorted, y_sorted, 
                    label=label, 
                    marker=marker, 
                    linestyle='-',
                    color=color,
                    alpha=0.9, # Slight transparency
                    markeredgecolor='white',
                    markeredgewidth=0.8)

        # --- Modern Styling Tweaks ---
        ax.set_xlabel("Pre-Train Accuracy over 95% (%)")
        ax.set_ylabel("Fine-tune Accuracy (%)")
        
        # Modern Legend: No frame, top-right or best location
        ax.legend(frameon=False, loc='best')
        
        # Remove extra whitespace
        plt.tight_layout()

        # Save
        file_path = Path(filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filename, bbox_inches='tight', transparent=True, pad_inches=0.01)
        print(f"Success: Plot saved to {filename}")

    except KeyError as e:
        print(f"Error: Missing expected column in data: {e}.")
    except Exception as e:
        print(f"Error during plotting: {e}")


# --- 3. Data Extraction ---

def extract_data(data):
    methods = ["Adam-NSCL", "Crisp_Hessian", "Crisp_KFAC", "Crisp_EKFAC", "Crisp_GN_Hessian"]
    data_update = {method: {"ft_accs": [], "pre_accs": [], "energy_threshold": []} for method in methods}
    
    for data_point in data:
        method = data_point.get('method')
        
        if method not in methods:
            continue
            
        if data_point['pre_accs'][-1] < 95:
            continue
            
        data_update[method]['energy_threshold'].append(data_point['energy_threshold'])
        data_update[method]['ft_accs'].append(data_point['ft_accs'][-1])
        data_update[method]['pre_accs'].append(data_point['pre_accs'][-1])

    data_update = {k: v for k, v in data_update.items() if v['pre_accs']}
    return data_update


# --- 4. Main Execution ---

if __name__ == "__main__":
    try:
        train_data_percentage = 0.15
        file_path = f'model_cache/fine_tuning_experiment_results_fc2_recalculation_sweep_{train_data_percentage}.pth'
        
        print(f"Loading data from: {file_path}")
        loaded_data = torch.load(file_path)
        
        if 'all_data' not in loaded_data:
            raise KeyError("'all_data' key not found in the loaded .pth file")
            
        raw_data = loaded_data['all_data']
        all_data = extract_data(raw_data)
        
        plot_metrics(all_data, filename="plots/roc_curve.pdf")
        
    except FileNotFoundError:
        print("Error: The specified .pth file was not found. Please check the path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")