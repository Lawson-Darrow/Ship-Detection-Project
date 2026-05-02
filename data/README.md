# Data Directory

Place the Kaggle Ships in Satellite Imagery dataset file here:

```text
data/shipsnet.json
```

The raw JSON file is intentionally not committed because it is larger than
GitHub's standard file-size limit. The main notebook can also load the same file
from Google Drive when running in Colab.

Expected file:

- `shipsnet.json`

Expected dataset:

- 4,000 labeled 80 x 80 RGB image chips
- 1,000 ship samples
- 3,000 no-ship samples
