# Week 4 Sentiment Analyzer - Analysis Report

## Baseline Model
I trained a `LogisticRegression` classifier using `CountVectorizer` (Bag-of-Words) on the NLTK `movie_reviews` dataset (2,000 total reviews: 1,000 positive, 1,000 negative) with an 80/20 train/test split. Because it only counts individual words without tracking word order, it works well on full-length movie reviews (**84.25% accuracy**, 337/400 correct: 164 TN, 36 FP, 27 FN, 173 TP).

## Challenge Set Results
Testing the model on 25 short, one-sentence reviews dropped accuracy to **56% (14/25)**:

* **3.1 Negation (60% - 3/5):** Missed `"not bad"` and `"did not hate"`.
* **3.2 Sarcasm (100% - 5/5):** False win—the model guessed `0` on every sentence, which matched all 5 negative labels.
* **3.3 Mixed Sentiment (60% - 3/5):** Missed sentences that started negative and turned positive after `"but"` or `"yet"`.
* **3.4 Intensity (0% - 0/5):** Guessed `0` on every positive sentence, even strong praise.
* **3.5 Domain Shift (60% - 3/5):** Missed positive pizza and running shoe reviews because those words aren't in movie reviews.

## Why It Broke & How to Fix It
1. **Short vs. Long Reviews:** The model trained on multi-paragraph essays, so feeding it 6-to-12-word sentences left almost the entire word-count table at zero and caused it to guess `0` (Negative) for all 25 test cases. Training on shorter reviews would fix this bias.
2. **No Word Order:** Counting single words misses `"not"` flipping `"bad"` or `"yet"` changing a sentence's direction. Checking word pairs (bigrams) or using a model that tracks word order would solve this.
3. **Movie-Only Vocabulary:** Training only on movie reviews left it blind to product and food terms, which requires broader training data.