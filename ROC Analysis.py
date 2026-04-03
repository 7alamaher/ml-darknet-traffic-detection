import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

rf_bin = trained_bin['Random Forest']
rf_mul = trained_mul['Random Forest']

# CASE 1: Binary ROC 
fig, ax = plt.subplots(figsize=(7, 6))

y_score_bin = rf_bin.predict_proba(X_test_bin)[:, 1]

# Map labels to 0/1 for roc_curve
y_test_bin_num = (y_test_bin == 'Darknet').astype(int)

fpr, tpr, _ = roc_curve(y_test_bin_num, y_score_bin)
roc_auc = auc(fpr, tpr)

ax.plot(fpr, tpr, color='steelblue', lw=2,
        label=f'Random Forest (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random guess')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.02])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve — Case 1: Binary (Benign vs Darknet)', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_case1_binary.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Case 1 AUC: {roc_auc:.4f}  |  Saved: roc_case1_binary.png")

# CASE 2: Multiclass ROC (One-vs-Rest) 
classes = ['Non-Tor', 'Non-VPN', 'Tor', 'VPN']
colors  = ['steelblue', 'darkorange', 'green', 'crimson']

# Binarize labels for One-vs-Rest
y_test_bin2 = label_binarize(y_test_mul, classes=classes)
y_score_mul = rf_mul.predict_proba(X_test_mul)

# Make sure column order matches 'classes'
class_order = list(rf_mul.classes_)
col_idx = [class_order.index(c) for c in classes]
y_score_mul = y_score_mul[:, col_idx]

fig, ax = plt.subplots(figsize=(8, 6))

for i, (cls, color) in enumerate(zip(classes, colors)):
    fpr_i, tpr_i, _ = roc_curve(y_test_bin2[:, i], y_score_mul[:, i])
    auc_i = auc(fpr_i, tpr_i)
    ax.plot(fpr_i, tpr_i, color=color, lw=2,
            label=f'{cls} (AUC = {auc_i:.4f})')

ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random guess')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.02])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve — Case 2: Multiclass (One-vs-Rest)', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_case2_multiclass.png', dpi=150, bbox_inches='tight')
plt.show()
print("Case 2 AUC per class:")
for i, cls in enumerate(classes):
    fpr_i, tpr_i, _ = roc_curve(y_test_bin2[:, i], y_score_mul[:, i])
    print(f"  {cls:<10} AUC = {auc(fpr_i, tpr_i):.4f}")
print("Saved: roc_case2_multiclass.png")

print("\nPhase 7 complete!")