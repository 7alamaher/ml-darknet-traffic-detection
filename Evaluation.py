## Phase 6 — Evaluation Metrics
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings('ignore')

# ── Helper: pick correct test set (MLP needs scaled data) 
def get_test_data(name, case):
    if name == 'MLP':
        return (X_test_bin_scaled, y_test_bin) if case == 1 else (X_test_mul_scaled, y_test_mul)
    return (X_test_bin, y_test_bin) if case == 1 else (X_test_mul, y_test_mul)

#  Helper: evaluate all models for one case
def evaluate_models(trained, case_num, case_name):
    print(f"\n{'='*65}")
    print(f"  {case_name}")
    print(f"{'='*65}")
    print(f"  {'Model':<20} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F-score':>9}")
    print(f"  {'-'*58}")

    results = {}
    for name, model in trained.items():
        X_test, y_test = get_test_data(name, case_num)
        y_pred = model.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        results[name] = {'accuracy': acc, 'precision': prec,
                         'recall': rec,  'f1': f1, 'y_pred': y_pred}

        print(f"  {name:<20} {acc:>8.4f}  {prec:>9.4f}  {rec:>7.4f}  {f1:>8.4f}")

    return results

#Evaluate both cases 
results_bin = evaluate_models(trained_bin, 1, "Case 1 — Binary (Benign vs Darknet)")
results_mul = evaluate_models(trained_mul, 2, "Case 2 — Multiclass (Tor/Non-Tor/VPN/Non-VPN)")

# ── Plot Confusion Matrices
def plot_confusion_matrices(trained, results, y_test, case_name):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    fig.suptitle(f'Confusion Matrices — {case_name}', fontsize=14, fontweight='bold')

    for i, (name, model) in enumerate(trained.items()):
        y_pred = results[name]['y_pred']
        cm = confusion_matrix(y_test, y_pred)
        
        # FIX: don't pass display_labels — let it use the cm directly
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=axes[i], colorbar=False, cmap='Blues')
        axes[i].set_title(f'{name}\nAcc: {results[name]["accuracy"]:.4f}',
                          fontsize=11)
        axes[i].tick_params(axis='x', rotation=30)

    axes[5].axis('off')
    plt.tight_layout()
    plt.savefig(f'confusion_matrices_{"binary" if "Binary" in case_name else "multi"}.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved!")

plot_confusion_matrices(trained_bin, results_bin, y_test_bin, "Binary")
plot_confusion_matrices(trained_mul, results_mul, y_test_mul, "Multiclass")

print("\nPhase 6 complete!")