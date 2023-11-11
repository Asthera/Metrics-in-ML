# Metrics-in-ML

Metrics are used to monitor and measure the performance of a model (during training and testing).

# Table of contents
1. [Сlassification](#Сlassification)
    1. [Accuracy](#Accuracy)
    2. [Precision](#Precision)
    3. [Recall](#Recall)
    4. [F1-Score](#F1-Score)
    5. [AUC-ROC](#AUC-ROC)
3. [Some paragraph](#paragraph1)
    1. [Sub paragraph](#subparagraph1)
4. [Another paragraph](#paragraph2)

## Classification metrics <a name="introduction"></a>
Metrics are used to monitor and measure the performance of a model (during training and testing), and don’t need to be differentiable. 

### Accuracy <a name="Accuracy"></a>
Measures the overall correctness of predictions.
Useful when classes are balanced, and you want a general sense of the model's performance.
```python 
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true, y_pred)
```

### Precision <a name="Precision"></a>
Measures the ability of the model to make correct positive predictions.
Useful when minimizing false positives is critical, such as in medical diagnoses or fraud detection.
```python 
from sklearn.metrics import precision_score
precision = precision_score(y_true, y_pred) # (True Positives) / (True Positives + False Positives)
```

### Recall (Sensitivity or True Positive Rate) <a name="Recall"></a>
Measures the model's ability to identify all actual positives.
Useful when minimizing false negatives is crucial, as in disease detection or search and rescue.
```python 
from sklearn.metrics import recall_score
recall = recall_score(y_true, y_pred) # (True Positives) / (True Positives + False Negatives)
```

### F1-Score <a name="F1-Score"></a>
Balances precision and recall, providing a single metric that considers both false positives and false negatives.
Useful when you want to strike a balance between precision and recall.
```python 
from sklearn.metrics import f1_score
f1 = f1_score(y_true, y_pred) # 2 * (Precision * Recall) / (Precision + Recall)
```
### Area Under the Receiver Operating Characteristic Curve <a name="AUC-ROC"></a>
Evaluates the model's ability to distinguish between positive and negative classes across different threshold values.
Useful when assessing the model's performance across different discrimination thresholds.

This metric is commonly used for binary classification problems with class probabilities.
```python 
from sklearn.metrics import roc_auc_score
auc_roc = roc_auc_score(y_true, y_scores)
```
<p align="center">
  <img src="https://github.com/Asthera/Metrics-in-ML/blob/main/sphx_glr_plot_roc_thumb.png" title="hover text">
</p>
