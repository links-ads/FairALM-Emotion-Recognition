import os
import argparse
import pandas as pd

from pathlib import Path
from mllm_emotion_classifier.utils.line_plot_separated import plot_fairness_vs_hparam
from EmoBox.EmoBox import EmoDataset

SENSITIVE_ATTR_DICT = {
    'iemocap': ['gender'],
    'cremad': ['gender', 'age', 'ethnicity'],
    'emovdb': ['gender'],
    'tess': ['agegroup'],
    'ravdess': ['gender'],
    'esd': ['gender'],
    'meld': ['gender'],
}

def parse_args():
    parser = argparse.ArgumentParser(description='Plot fairness metrics vs hyperparameter')
    parser.add_argument('--hparam', type=str, default='temperature', choices=['temperature', 'top_p'],
                        help='Hyperparameter to plot (temperature or top_p)')
    parser.add_argument('--dataset', type=str, default='meld', 
                        choices=['iemocap', 'cremad', 'emovdb', 'ravdess', 'meld'],
                        help='Dataset name')
    parser.add_argument('--fold', type=int, default=None,
                        help='Fold number (default: None, aggregates all folds)')
    parser.add_argument('--model', type=str, default='qwen2-audio-instruct',
                        help='Model name')
    parser.add_argument('--show-std', action='store_true', default=True,
                        help='Show standard deviation in plots')
    return parser.parse_args()

def main():
    args = parse_args()
    
    dataset = args.dataset
    hparam = args.hparam
    fold = args.fold
    model = args.model
    sensitive_attrs = SENSITIVE_ATTR_DICT[dataset]
    
    metadata_dir = Path('EmoBox/data/')
    dataset_path = metadata_dir / dataset
    n_folds = len([d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith("fold_")]) if dataset != 'esd' else 2
    out_dir = Path('outputs') / "temperature_runs" if hparam == 'temperature' else Path('outputs') / "topp_runs"
    
    if fold is None:
        dfs = []
        for f in range(1, n_folds + 1):
            results_csv = out_dir / model / dataset / f'fold_{f}.csv'
            df_fold = pd.read_csv(results_csv)
            dfs.append(df_fold)
        df = pd.concat(dfs, ignore_index=True)
    else:
        results_csv = out_dir / model / dataset / f'fold_{fold}.csv'
        df = pd.read_csv(results_csv)
    
    print(f"{len(df)} rows loaded")
    print(df.head(5))
    
    # Plot fairness metrics
    fold_label = fold if fold is not None else 'all'
    os.makedirs(out_dir / 'figures', exist_ok=True)
    outpath = out_dir / 'figures' / f'{dataset}_line_fold_{fold_label}.png'
    
    plot_fairness_vs_hparam(
        df,
        hparam,
        ['statistical_parity', 'equal_opportunity', 'overall_accuracy_equality'],
        sensitive_attrs,
        model,
        dataset,
        fold,
        show_std=args.show_std,
        output_path=outpath,
    )
    
    print(f"Plot saved to {outpath}")

if __name__ == "__main__":
    main()