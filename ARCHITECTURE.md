# Architecture: spoken grammar score (SHL Hiring Assessment 2026)

## Task
Predict a 0 to 5 grammar score for a 45 to 60 second spoken answer. The label is a human mean opinion score. Scoring uses RMSE and Pearson correlation on a hidden test set of 216 clips.

## Final result
`submission.csv`, from the model below: **public RMSE 0.4247**. Public leaderboard leaders score about 0.31 to 0.33.

## Diagram

```
                       spoken answer (WAV, 16 kHz)
                                 │
            ┌────────────────────┴────────────────────┐
            ▼                                         ▼
  Whisper-small.en transcript              WavLM-base-plus (frozen)
  + word confidence                        hidden states, layers 8, 9, 10
            │                                         │
            ▼                                         ▼ mean over time, concatenated
  Text features (61)                       PCA → 64 components (fit on train only)
  · grammar errors (LanguageTool)                       │
  · fluency (rate, pauses, fillers)                     │
  · vocabulary (diversity, repetition)                  │
            │                                         │
            └──────────────┬──────────────────────────┘
                           ▼
                 125 input features per clip
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
    Ridge regression              Gradient boosting
    (linear, regularised)         (non-linear)
            │                             │
            └──────────────┬──────────────┘
                           ▼
          non-negative blend (weights from out-of-fold predictions)
                           │
                           ▼
          affine recalibration  y = 1.094·x − 0.307   (counters shrinkage)
                           │
                           ▼
                clip to [0, 5]  →  submission
```

## Decisions and why

**1. Text features (61 values).** They measure things a rater would notice: grammatical errors, pauses and fillers, and vocabulary range. Each feature can be explained in one sentence. Whisper-small was used because it runs on CPU in reasonable time, and its transcripts include word timings, which the fluency features need.

**2. WavLM-base, layers 8 to 10 (64 values after PCA).** Audio carries information about delivery that a transcript loses. WavLM is a self-supervised speech model used as a frozen feature extractor, so no training is needed on 769 clips. Middle layers were chosen because the final layer is specialised for the model's own pretraining task. WavLM-base is cheaper to run than the large version, and the large version did not do better in our validation (see below).

**3. PCA to 64 components, fitted on training clips only.** Three layers of 768 values each would overfit on 769 clips. PCA keeps the main variation. Fitting it on training clips only means no test information leaks into the features.

**4. Ridge regression and gradient boosting, blended.** Ridge is a linear model that is hard to overfit and easy to explain. Gradient boosting can capture non-linear effects that a straight line misses. The blend was better than either model alone on speaker-grouped validation.

**5. Non-negative blend weights.** Each model's contribution has to be positive, so the blend stays interpretable. Negative weights could cancel one model against the other and are hard to justify.

**6. Affine recalibration.** Regression predictions shrink toward the average, so the correction stretches them back out. It is fitted on out-of-fold predictions, so it does not use the clips it is scored on. Its effect on the leaderboard was small, so it is kept for its logic rather than its score.

**7. Clipping to [0, 5].** The scale is defined on that range, and a score outside it is meaningless.

**8. Zero-label clips kept in training.** The brief allows 0 as a valid score, so removing those clips would discard examples the test set may contain.

**9. No speech-detection rule in the submission.** The idea is logically sound (a clip with no speech cannot be graded), but it was not validated before the deadline, so it is not in the model.

## Why the score is limited

Our model uses fewer and smaller inputs than the leaderboard leaders. The specific limits in our work:

- **Smaller models for the main features.** The large WavLM model scored worse than WavLM-base under speaker-grouped validation (0.82 against 0.77). The large Whisper encoder helped only when combined at one layer, which looked like noise.
- **No word timestamps on the large-model transcripts.** The large-model transcripts we produced have no word timing, so the fluency features that depend on it cannot be computed. Without fluency, those transcripts scored worse than the local ones (0.80 against 0.78 under speaker-grouped validation).
- **No rubric-based language-model features.** Running a language model with the scoring rubric needs GPU time we did not have.
- **Fewer base models.** The blend has two models, which leaves less room to correct individual errors.
- **Distribution shift.** Test clips are shorter than training clips (median 45 s against 60 s). Validation on training clips may therefore be tuned to longer clips.
- **Validation gap.** Random-fold validation gives about 0.58 RMSE, speaker-grouped validation about 0.78, and the leaderboard 0.42. Random folds overstate how well the model generalises to new speakers.

Choices were made to keep the model explainable and to avoid tuning to the leaderboard. That is why the score is lower than the leaders', and it is the trade-off this submission makes.

## Validation

- **Random stratified CV** (5 folds, 2 repeats): about 0.58 RMSE. Optimistic, because clips from the same speaker can fall into both training and validation folds.
- **Speaker-grouped CV** (speakers approximated by clustering): about 0.78 RMSE. The more honest estimate for new speakers.
- **Leaderboard:** 0.4247.

## Things tested and not used

Each of these was tried to improve the score. None helped enough to keep:
- A blend of the current model with separate large-feature models. It scored 0.4234 publicly, but mixing two unrelated models has no clear basis.
- WavLM-large in place of WavLM-base. Worse under speaker-grouped validation.
- Chunked WavLM-large features, and the Whisper encoder. No gain in layer sweeps.
- Large-model verbatim transcripts, as a full replacement or as extra measures. Worse.
- Reweighting training clips toward the test set. No gain on short clips.
- Removing duration from the embeddings, and joint PCA with test clips. No gain.
- XGBoost in place of gradient boosting, and tuned settings. No gain beyond noise.
- Speaker pooling of predictions. Pearson improved slightly on training, but the leaderboard score was slightly worse (0.4274).
- A single model chosen on speaker-grouped validation. Scored 0.5127 publicly.
- Fold schemes based on duration. No change in error estimates.

## Known limitations

1. Our inputs are fewer and smaller than the leaders' (see above).
2. Random-fold validation overstates performance because of speaker overlap.
3. Test clips differ from training clips in length, and the audio features partly separate the two sets.
4. Zero-label clips are not treated specially.
5. The blend weights and recalibration come from random folds. Speaker-grouped folds would give different values, and on the leaderboard the grouped-fold choices scored worse.
