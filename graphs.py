import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, RocCurveDisplay
from sklearn.preprocessing import StandardScaler

def save_graphs(model, feature_names, X_test_scaled, Y_test, static_dir):
    static_dir.mkdir(exist_ok=True)

    # Graph 1: top 12 features by importance
    coefs = pd.Series(model.coef_[0], index=feature_names)
    top = coefs.reindex(coefs.abs().sort_values(ascending=False).head(12).index)
    top = top.sort_values()

    plt.figure(figsize=(8, 5))
    plt.barh(top.index, top.values,
             color=["tomato" if v > 0 else "steelblue" for v in top.values])
    plt.title("What affects placement (red = helps, blue = hurts)")
    plt.xlabel("Coefficient")
    plt.savefig(static_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Graph 2: ROC curve
    RocCurveDisplay.from_estimator(model, X_test_scaled, Y_test)
    plt.plot([0, 1], [0, 1], "k--", label="Random guess")
    plt.title("ROC curve")
    plt.legend()
    plt.savefig(static_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.grid()
    plt.close()

   
    
     