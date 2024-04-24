# Micro-EMA
This repository ...

## Data description
The data is taken from [here](https://osf.io/4zajm)
- processed.csv: each row contains features from 1 minute intervals
- labeledfeatures.csv: same as processed.csv but it has details of the micro-ema responses as well
- annotations.csv: 21 rows containing details about the activities and their corresponding timestamps
- elec.csv: contains the raw data (timestamp, voltage)
- In labeledfeatures.csv the ema responses are given but they there are no entries for TSST rest and cry rest. The ema responses are the same for all the cry activities

## Additional Details
- Removed the following from In-lab dataset
  - removed P9 because the data doesn't mention event type
  - removed p18 because `labeledfeatures.csv` not present
- Removed the following from In-Wild Dataset
  - P101, P107, P111, P112, P113, P114 were removed because of negative Sample voltage.

## Setting up the environment
```bash
pipenv -r install requirements.txt
```

## Candidate EMA selection:

- To select the candidate EMAs that have high biserial-correlation with intended stressors.

```python
python ema_stressors_correlation.py
```

- The correlation plot of all EMAs with intended stressors is saved at `/Plots/Correlation_ema_intended.png`

## Training the models:

### Traditional Machine Learning:
- Run `train_chunk_features.ipynb`
- This notebook creates the following hand-crafted features:

| Signal | Features |
|--------|----------|
|Heart-rate Variability| Low freq. energy (0.1-0.2Hz),  RMSSD, SDSD|
|Non-heart-rate variability| Mean, Mode, Minimum, Range, $40^{th}$, $60^{th}$, $80^{th}$ Percentile.

- These features are then used to train SVM, Decision Tree, Gradient Boosting, and AdaBoost.

- The results after validating the models using LOSO are written to `Results/loso_chunks.csv`.

### Deep Learning:
#### Using LOSO
- To run the tensorflow model and validate it using LOSO
```python
python train_chunk_loso.py
```

#### Without LOSO
- To train the model on entire In-Lab data
```python
python train_chunk.py
```
- The best model checkpoint will be written to `chkpts/v4(2)`

## Testing the model
- Run `testing_inwild.ipynb`.
- This notebook generates accuracy plots for top 3 EMAs obtained after EMA selection over varying window lengths.
- All the plots are saved at `Plots/In-Wild`
