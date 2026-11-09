# Assignment 3: Time-Series Data (Jena Climate Temperature Forecasting)

**Course:** Advanced Machine Learning (64061)
**Author:** Shujath Ali Ansari Mohammed

This assignment applies recurrent neural networks (Python, Keras 3 / TensorFlow) to the Jena weather time series from class. Each model uses the previous 5 days of hourly data (14 variables) to predict the temperature 24 hours ahead. The experiments test the three methods listed in the assignment: the number of units in the stacked recurrent layers, LSTM instead of GRU, and a 1D convnet combined with an RNN. A follow-up tests stronger dropout. The best models, chosen on validation MAE and re-trained with a second seed, were then run on the test set.

## Files

| File | What it is |
|---|---|
| `Assignment3_Summary.docx` / `.pdf` | **Summary report** (start here): results tables, 7 graphs and conclusions |
| `Assignment3_TimeSeries.ipynb` | Executed Jupyter notebook (run on Kaggle, GPU T4) with all code and outputs |
| `Assignment3_TimeSeries.html` | HTML export of the executed notebook |

## Key result

All 9 recurrent models in Experiments A–C beat the common-sense baseline on validation (2.443 °C). The stronger-dropout model in Experiment D did not (2.473 °C). The best model on validation was a **stacked GRU with 32 units per layer** (validation MAE 2.333 °C, mean of 2 seeds). It reached a **test MAE of 2.535 ± 0.035 °C**, compared with 2.621 °C for the baseline. Larger networks (64–128 units), LSTM layers, a Conv1D front end and stronger dropout did not improve on it.

## How to reproduce

Open the notebook on Kaggle, attach the dataset `mnassrib/jena-climate` (or let the notebook download the original file used in class), turn on a GPU accelerator and choose **Run All**. A full run takes about one hour on a T4 GPU.
