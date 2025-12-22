import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def plot_global_over_attribute_values(df, top_p, attribute, attribute_values, temperature=False, metric='f1_macro', 
                                  model=None, dataset=None, fold=None):
    """
    Plot bar comparison of a single global metric between Male and Female.
    
    Args:
        df: DataFrame with gender-specific metrics
        metric: global metric name to compare ('f1_macro', 'f1_weighted', 'accuracy_unweighted', 'accuracy_weighted')
        model: model name for title
        dataset: dataset name for title
        fold: fold number for title
    """
    if temperature:
        df = df[df['temperature'] == top_p]
    else:
        df = df[df['top_p'] == top_p]
    
    assert metric in ['f1_macro', 'f1_weighted', 'accuracy_unweighted', 'accuracy_weighted'], "Invalid metric"
    
    attribute_values = sorted(list(attribute_values))
    
    cmap = cm.get_cmap('tab10')
    colors = {e: cmap(i % 10) for i, e in enumerate(attribute_values)}
    
    data = []
    for value in attribute_values:
        col_name = f'{attribute}_{value}_global_{metric}'
        if col_name in df.columns:
            data.append(df[col_name].mean())
        else:
            data.append(0)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    x = np.arange(len(attribute_values))
    width = 0.6
    
    bars = ax.bar(x, data, width, color=[colors[g] for g in attribute_values], alpha=0.8)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    metric_label = metric.replace('_', ' ').title()
    ax.set_ylabel(metric_label, fontsize=14, fontweight='bold')
    ax.set_title(f'{metric_label} Comparison by {attribute.title()}', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(attribute_values, fontsize=14)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.tick_params(axis='both', labelsize=12)
    
    # if model and dataset:
    #     fold_str = fold if fold is not None else 'All'
    #     plt.suptitle(f'{model} on {dataset} Fold {fold_str}', fontsize=12, y=0.98)
    
    plt.tight_layout()
    plt.show()


def plot_classwise_over_attribute_values(df, top_p, emotions, attribute, attribute_values, temperature=False, metric='accuracy', 
                                      model=None, dataset=None, fold=None):
    """
    Plot bar comparison of emotion-specific metrics between Male and Female.
    
    Args:
        df: DataFrame with gender-specific metrics by emotion
        emotions: list of emotion names
        metric: one of 'accuracy', 'true_positive_rate', 'false_positive_rate', 'f1_score'
        model: model name for title
        dataset: dataset name for title
        fold: fold number for title
    """
    if temperature:
        df = df[df['temperature'] == top_p]
    else:
        df = df[df['top_p'] == top_p]

    assert metric in ['accuracy', 'true_positive_rate', 'false_positive_rate', 'f1_score'], "Invalid metric"

    cmap = cm.get_cmap('tab10')
    attribute_values = sorted(list(attribute_values))
    colors = {e: cmap(i % 10) for i, e in enumerate(attribute_values)}
    
    emotions_list = sorted(list(emotions))
    data = {value: [] for value in attribute_values}
    
    for emotion in emotions_list:
        for value in attribute_values:
            col_name = f'{attribute}_{value}_classwise_{metric}_{emotion}'
            if col_name in df.columns:
                data[value].append(df[col_name].mean())
            else:
                data[value].append(0)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(emotions_list))
    n_values = len(attribute_values)
    width = 0.8 / n_values
    
    offset = width * (n_values - 1) / 2
    
    for i, value in enumerate(attribute_values):
        bar_positions = x - offset + i * width
        bars = ax.bar(bar_positions, data[value], width, label=value, 
                     color=colors[value], alpha=0.8)
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    metric_label = metric.replace('_', ' ').title()
    ax.set_xlabel('Emotion', fontsize=18, fontweight='bold')
    ax.set_ylabel(metric_label, fontsize=18, fontweight='bold')
    ax.set_title(f'{metric_label} Comparison by Emotion and {attribute.title()}', 
                 fontsize=20, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(emotions_list, rotation=15, ha='right', fontsize=16)
    ax.set_ylim(0, 1.1)  # Increased to accommodate labels
    ax.legend(fontsize=14, title=attribute.title(), title_fontsize=16)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.tick_params(axis='both', labelsize=16)
    
    # if model and dataset:
    #     fold_str = fold if fold is not None else 'All'
    #     plt.suptitle(f'{model} on {dataset} Fold {fold_str}', fontsize=14, y=0.98)
    
    plt.tight_layout()
    plt.show()