"""
Utility for evaluation metrics for classification models.
"""
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


def classification_metrics_full(y_true, y_pred):
    """
    Compute a comprehensive set of classification metrics.
    """
    
    # Aggregated metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", 
                                                                                 zero_division=0)
    # Per-class metrics
    precision_c, recall_c, f1_c, support = precision_recall_fscore_support(y_true, y_pred, average=None, 
                                                                           zero_division=0)
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    results = {
        "aggregated":{
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro
        },
        "per_class": {
            "precision": precision_c,
            "recall": recall_c,
            "f1": f1_c,
            "support": support,
        },
        "confusion_matrix": cm,
    }

    return results