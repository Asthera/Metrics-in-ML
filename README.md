# Metrics-in-ML

Metrics are used to monitor and measure the performance of a model (during training and testing).

<details>
  <summary>Classification</summary>

  1. [Accuracy](#Accuracy)
  2. [Precision](#Precision)
  3. [Recall](#Recall)
  4. [F1-Score](#F1-Score)
  5. [AUC-ROC](#AUC-ROC)
  6. [Confusion Matrix](#ConfusionMatrix)
</details>

<details>
  <summary>Regression</summary>


</details>

<details>
  <summary>TODO</summary>
  
  1. [TODO](#Regression)
x
  
</details>





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


### Confusion Matrix <a name="ConfusionMatrix"></a>
A table that describes the performance of a classification algorithm.
Provides a detailed breakdown of true positives, true negatives, false positives, and false negatives.

Useful for understanding where the model excels and where it falls short.
```python 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
```
<p align="center">
  <img src="https://github.com/Asthera/Metrics-in-ML/blob/main/conf_matrix.jpg" title="hover text">
</p>


## TODO <a name="TODO"></a>
Maked for:
- [X] Classification
- [ ] Regression
- [ ] Segmentation
- [ ] Object Detection



## Inspiraion

Classification: Accuracy, Precision, Recall, F-1 score. I would probably use precision and recall rather than accuracy.
Regression: Mean Square error, Root Mean Square Error, Mean Absolute Error.
Segmentation: Mean Intersection over Union (IOU), Dice Coefficient, Pixel Accuracy.
Object Detection: Average Precision (AP) and Mean Average Precision (mAP)
Pose Estimation: Percentage of Detected Joints, Object Keypoint Similarity, Mean Per Joint Position Error.


