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
- removed P9 because the data doesn't mention event type
- removed p18 because `labeledfeatures.csv` not present
