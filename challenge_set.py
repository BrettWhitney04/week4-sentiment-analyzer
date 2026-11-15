from sentiment_analyzer import preprocess_text
from sentiment_analyzer import train_and_evaluate 
from sklearn.metrics import accuracy_score


def run_challenge():
    vectorizer, model = train_and_evaluate()
    category_scores = {}
    total_correct = 0
    total_reviews = 0

    for category, items in CHALLENGE_DATA.items():
        texts = [preprocess_text(text) for text, label in items]
        y_true = [label for text, label in items]
        X_vec = vectorizer.transform(texts)
        y_pred = model.predict(X_vec)

        acc = accuracy_score(y_true, y_pred)
        category_scores[category] = acc
        total_correct += sum(y_pred == y_true)
        total_reviews += len(items)
        print(f"\n---{category} ({acc:.0%}) ---")
        for (raw_text, true_val), pred_val in zip(items,y_pred):
            print(f"True Value: {true_val} | Predicted Value: {pred_val} | Raw Text: {raw_text}")
    print(total_correct/total_reviews)
    worst = min(category_scores, key=category_scores.get)
    print (worst)


CHALLENGE_DATA = {
    "3.1 Negation": [
        ("The acting was not good at all.", 0),
        ("I wasn't impressed by the plot or the pacing.", 0),
        ("This isn't a film I'd recommend to anyone.", 0),
        ("The ending was not bad, actually quite satisfying.", 1),
        ("I did not hate this movie, it was genuinely fun.", 1),
    ],
    "3.2 Sarcasm": [
        ("Oh wonderful, another two hours of my life I'll never get back.", 0),
        ("Brilliant script—if your favorite genre is watching paint dry.", 0),
        ("Truly a masterpiece of boredom; I slept like a baby.", 0),
        ("Bravo to the director for making a thriller with zero thrills.", 0),
        ("What a fantastic waste of a perfectly good Friday evening.", 0),
    ],
    "3.3 Mixed Sentiment": [
        ("Beautiful cinematography, completely wasted on a terrible, incoherent script.", 0),
        ("The soundtrack is gorgeous, though the characters are flat and forgettable.", 0),
        ("Rough first act and clunky dialogue, but the finale is pure magic.", 1),
        ("Weak villain, yet the lead performances make this an absolute joy to watch.", 1),
        ("Stunning visual effects cannot save a story this dull and predictable.", 0),
    ],
    "3.4 Intensity": [
        ("It was fine, nothing special.", 1),
        ("A decent enough movie for a rainy afternoon.", 1),
        ("Quite good, I enjoyed most of it.", 1),
        ("Really great filmmaking with strong performances.", 1),
        ("An extraordinary, unforgettable triumph of cinema.", 1),
    ],
    "3.5 Domain Shift": [
        ("The blender motor burned out after making two smoothies.", 0),
        ("Crispy crust, fresh basil, and rich marinara—best pizza in town.", 1),
        ("Battery drains in an hour and the charging port feels loose.", 0),
        ("Overpriced steak arrived cold and chewy; service was nonexistent.", 0),
        ("Lightweight running shoes with incredible arch support right out of the box.", 1),
    ],
}

if __name__ == "__main__":
    run_challenge()
