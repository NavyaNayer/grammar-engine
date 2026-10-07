# Spoken grammar scoring: final submission

Predicts a grammar score from 0 to 5 for a 45 to 60 second spoken answer, for the SHL Hiring Assessment 2026 Kaggle challenge. The metric is RMSE, with Pearson correlation also reported.

## Result

| Measure | Value |
|---|---|
| Public leaderboard RMSE | **0.4247** |
| Cross-validated RMSE, random folds | about 0.58 (optimistic: speakers can appear in both training and validation folds) |
| Cross-validated RMSE, speaker-grouped folds | about 0.78 (the more honest estimate) |

The leaderboard result sits between the two estimates. The architecture and the reasons for each choice are in [ARCHITECTURE.md](ARCHITECTURE.md).

## Approach in brief

1. **Text features (61 values).** Whisper-small.en transcripts, with grammar-error rates from LanguageTool, fluency measures, and lexical diversity.
2. **Audio features (64 values).** Frozen WavLM-base-plus, layers 8 to 10, mean-pooled over time and reduced to 64 principal components.
3. **Regression.** Ridge regression and gradient boosting, combined with non-negative weights fitted on out-of-fold predictions.
4. **Recalibration.** A linear correction fitted on out-of-fold predictions, to counter the shrinkage typical of regression.
5. **Output.** Predictions clipped to [0, 5].

## Repository contents

```
submission.csv              final predictions (216 test clips)
ARCHITECTURE.md             architecture, validation and limitations
requirements.txt            Python dependencies
src/grammar_scoring/        the feature, model and submission code
scripts/reproduce_final.py  rebuilds submission.csv from the raw data
```

## Reproducing the submission

The competition data is not included, so you need to download it from Kaggle first. Put the folder containing `train.csv`, `test.csv`, `train/` and `test/` anywhere you like, then run:

```bash
pip install -r requirements.txt
python scripts/reproduce_final.py --data-dir path/to/data/raw --cache-dir .cache
```

The first run extracts WavLM features for all 985 clips, which takes about an hour on a CPU. Later runs reuse the cache. The script writes `reproduced_submission.csv` and reports how closely it matches `submission.csv`. In our run the correlation was 0.9996.

## Limitations

- The test clips are shorter than the training clips (median 45 s against 60 s). The model may partly rely on patterns common in longer clips.
- Random cross-validation overstates performance, because the same speakers appear in several clips.
- Clips with no speech are not treated specially. A no-speech rule was tested but not validated, so it is not part of this submission.
- Several model choices were compared on the same validation data, so the reported numbers are slightly optimistic.
