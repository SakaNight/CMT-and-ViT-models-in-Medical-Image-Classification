import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os

def process_excel(file_path):
    df = pd.read_excel(file_path)
    return df

df = process_excel('your_local_folder/result1.xlsx')

def create_plot(data, y_column, title, filename, y_label, y_lim=None):
    plt.figure(figsize=(10, 4))
    g = sns.catplot(
        data=data, 
        kind="bar",
        x="Model", y=y_column, hue="Attention", col="Dataset",
        palette="muted", height=3.5, aspect=1.2, legend_out=False
    )

    g.set_axis_labels("Model", y_label)
    g.set_titles("{col_name} Dataset", fontsize=10)
    g.fig.suptitle(title, fontsize=12)
    g.fig.subplots_adjust(top=0.85, wspace=0.1)

    if y_lim:
        g.set(ylim=y_lim)

    for ax in g.axes.flat:
        ax.tick_params(labelsize=8)
        ticks = ax.get_xticks()
        labels = [item.get_text() for item in ax.get_xticklabels()]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=45, ha='right')

    g.add_legend(title="Attention Mechanism", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()

    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"The plot has been generated and saved as '{filename}'.")

create_plot(df, "Accuracy", 
            "Accuracy Comparison of Different Models and Attention Mechanisms", 
            "accuracy_comparison_compact.png", 
            "Accuracy", 
            y_lim=(0, 0.55))

create_plot(df, "F1", 
            "F1 Score Comparison of Different Models and Attention Mechanisms", 
            "f1_comparison_compact.png", 
            "F1 Score", 
            y_lim=(0, 0.55))

create_plot(df, "Throughput", 
            "Throughput Comparison of Different Models and Attention Mechanisms", 
            "throughput_comparison_compact.png", 
            "Throughput (images/second)")