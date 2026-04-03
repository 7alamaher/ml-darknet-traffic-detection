
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

rf_bin = trained_bin['Random Forest']
rf_mul = trained_mul['Random Forest']

feature_names = X_train_bin.columns.tolist()

def plot_feature_importance(model, feature_names, case_name, filename, top_n=15):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    top_features = [feature_names[i] for i in indices]
    top_values   = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top_features[::-1], top_values[::-1],
                   color='steelblue', edgecolor='white')

    # Add value labels on bars
    for bar, val in zip(bars, top_values[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)

    ax.set_xlabel('Feature Importance Score', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importances — {case_name}',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {filename}")

    # Print ranked list
    print(f"\nTop {top_n} features for {case_name}:")
    for rank, (feat, val) in enumerate(zip(top_features, top_values), 1):
        print(f"  {rank:>2}. {feat:<35} {val:.4f}")

    return top_features, top_values

#  Case 1: Binary
top_feat_bin, top_val_bin = plot_feature_importance(
    rf_bin, feature_names,
    "Case 1: Binary (Benign vs Darknet)",
    "feature_importance_case1.png"
)

#  Case 2: Multiclass
top_feat_mul, top_val_mul = plot_feature_importance(
    rf_mul, feature_names,
    "Case 2: Multiclass (Tor/Non-Tor/VPN/Non-VPN)",
    "feature_importance_case2.png"
)

# Side-by-side comparison of top 5
print("\n" + "="*55)
print("  Top 5 Feature Comparison")
print("="*55)
print(f"  {'Rank':<6} {'Case 1':<35} {'Case 2'}")
print(f"  {'-'*53}")
for i in range(5):
    print(f"  {i+1:<6} {top_feat_bin[i]:<35} {top_feat_mul[i]}")

print("\nPhase 8 complete!")