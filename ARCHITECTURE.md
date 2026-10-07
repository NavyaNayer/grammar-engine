# Final architecture: spoken grammar score (SHL Hiring Assessment 2026)

## Task
Predict a 0 to 5 grammar score for a 45 to 60 second spoken answer. The label is a human mean opinion score. Metrics are RMSE and Pearson on a hidden test set (216 clips). The public leaderboard scores about 60% of it, and the private score covers the rest.

## Final submission
`FINAL_submission.csv`, from the model below. Public RMSE **0.4247**.

## Model

Each step has a reason for being there.

**1. Text features (61 values).** Computed from a Whisper-small.en transcript, with per-word confidence.
- Grammar checks: LanguageTool error rates per 100 words, clean-sentence ratio, and error types. These measure grammatical errors directly.
- Fluency: speaking rate, pause statistics from the audio, filler and false-start rates, and transcript confidence. These measure how smoothly the answer is delivered.
- Lexical: vocabulary diversity (moving-average type-token ratio), repetition, and sentence length. These measure range and variety of language.

**2. Audio features (64 values).** Frozen WavLM-base-plus, layers 8, 9 and 10, mean-pooled over time, on the first 65 seconds of each clip. The three layers are concatenated and reduced to 64 principal components, with the PCA fitted on training clips only. Middle layers were chosen because they carry more speech-delivery information than the final, recognition-tuned layer. The choice came from a layer sweep under random CV.

**3. Regression.** Two standard models on the combined features:
- Ridge regression: a linear model with regularisation, easy to interpret.
- Gradient boosting: captures non-linear effects that a linear model would miss.

Their predictions are combined with non-negative weights fitted on out-of-fold predictions (ridge 0.38, gradient boosting 0.62). Non-negative weights keep the combination interpretable: each model's contribution is positive and sums to one.

**4. Recalibration.** A linear correction, `y = 1.094 x - 0.307`, fitted on out-of-fold predictions. Regression predictions shrink toward the average, so this stretches them back out. The correction is fitted on out-of-fold predictions and checked with a nested split, so it doesn't use the scored clips.

**5. Output.** Predictions clipped to [0, 5]. All training clips are kept, including the 37 with a label of 0. The brief allows 0 as a valid score.

**Reproduction note.** The submission came from the original run, with weights 0.38 and 0.62 and the correction 1.094 and -0.307. Rebuilding with `scripts/reproduce_final.py` gives weights 0.33 and 0.67 and the correction 1.073 and -0.244, because the fold split and the subsample differ slightly between runs. The rebuilt predictions agree with the submitted file (correlation 0.9996, mean absolute difference 0.023), so the difference is small, but the exact numbers depend on the run.

## Training and validation
- Principal components and regression weights are fitted on training clips only.
- Validation uses 5-fold stratified cross-validation, repeated twice. In this version the folds are random, which lets clips from the same speaker appear in training and validation. That makes the cross-validation estimate optimistic (about 0.58 RMSE).
- Speaker-grouped cross-validation gives a more honest estimate for the same features (about 0.78 RMSE). The leaderboard result (0.4247) sits between the two.

## Limitations
1. The test clips are shorter than the training clips (median 45 s against 60 s), and the audio features separate the two sets. The model may partly fit patterns common in longer clips.
2. Random cross-validation overstates performance because of speaker overlap.
3. Recalibration gained nothing measurable on the leaderboard (0.4248 to 0.4247), so it's kept for its logic, not its score.
4. Zero-label clips are not treated specially.

## Optional step, pending validation
A no-speech rule would set the score to 0 for clips where a speech model hears no words. It's logically justified, since a clip with no speech can't be graded for grammar. It is not in the final file yet. It will be added only if it catches most training zeros without zeroing real answers.

## Dropped (tested, not used)
These were tried to improve the score. None had a clear justification or measurable gain, so they are not part of the architecture:
- A blend of the current model with separate GPU-feature models (80/20). It scored 0.4234 publicly, but mixing two unrelated models has no principled basis.
- WavLM-large in place of WavLM-base. Worse under grouped CV (0.82 against 0.77).
- The Colab chunk features and the Whisper encoder features. No gain in layer sweeps.
- Colab verbatim transcripts. Worse as a replacement and no gain as added measures.
- Reweighting training clips toward the test set. No gain on short clips.
- Removing duration from the embeddings, and joint PCA with test clips. No gain.
- A stretch factor around the mean, and a single-model variant chosen by grouped CV. The single model scored 0.5127 publicly.
