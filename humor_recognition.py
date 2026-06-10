from transformers import pipeline
from datasets import load_dataset

# loading ColBERT Humor Detection dataset
print("Lade Dataset...")
dataset = load_dataset("CreativeLang/ColBERT_Humor_Detection", split="train[:200]")

print(dataset.column_names)
print(dataset[0])

# load classifier
print("Lade Classifier...")
classifier = pipeline("zero-shot-classification", model="cross-encoder/nli-MiniLM2-L6-H768")

# classification
labels = ["humorous", "not humorous"]
results = []

print("Klassifiziere Texte...")
for i, example in enumerate(dataset):
    text = example["text"]
    label_true = "humorous" if example["humor"] == 1 else "not humorous"
    
    output = classifier(text, candidate_labels=labels)
    label_pred = output["labels"][0]
    
    results.append({
        "text": text,
        "true": label_true,
        "predicted": label_pred,
        "correct": label_true == label_pred
    })
    
    if i % 20 == 0:
        print(f"  {i}/200 done...")

# results
correct = sum(r["correct"] for r in results)
print(f"\nAccuracy: {correct}/200 = {correct/200:.2%}")

# show errors
print("\n--- FALSCH KLASSIFIZIERT ---")
errors = [r for r in results if not r["correct"]]
for r in errors[:20]:
    print(f"[True: {r['true']}] [Pred: {r['predicted']}]")
    print(f"  {r['text']}\n")