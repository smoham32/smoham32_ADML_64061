# Assignment 1: Neural Networks (IMDB Review Classification)

**Course:** Advanced Machine Learning (64061)
**Author:** Shujath Ali Ansari Mohammed

This assignment tunes the IMDB sentiment-classification network from class (R, Keras 3 / TensorFlow). It tests how the number of hidden layers, the number of hidden units, the mse loss, the tanh activation, and regularization (L2 and dropout) affect validation and test accuracy.

## Files

| File | What it is |
|---|---|
| `Assignment1_Report_Shujath_Ali_Ansari_Mohammed.pdf` | **Summary report** (start here): findings, graphs, conclusions and recommendations, without code |
| `Assignment1_Code.Rmd` | R Markdown source with all code |
| `Assignment1_Code.html` | Knitted output of the Rmd: all code, tables and graphs |
| `results/experiment_results.rds` | Saved training results (18 configurations × 3 seeds), reused when the Rmd is re-knitted |

## Key result

The final model (2 hidden layers × 32 units, relu, binary cross-entropy) reached **89.04% validation accuracy**, compared with 88.82% for the class baseline. After retraining on all 25,000 training reviews, both models scored about **88.3% on the test set**.

## How to reproduce

Open `Assignment1_Code.Rmd` in RStudio and click **Knit**. If `results/experiment_results.rds` is present, the saved results are reused. To retrain every model (slow), set `rerun: true` in the YAML header.
