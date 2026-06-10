from transformers import pipeline
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# load the dataset
print("Lade Dataset...")
dataset = load_dataset("CreativeLang/ColBERT_Humor_Detection", split="train[:200]")

# load the classifier
print("Lade Classifier...")
classifier = pipeline("zero-shot-classification", model="cross-encoder/nli-MiniLM2-L6-H768")

# classification
labels = ["humorous", "not humorous"]
results = []

print("Klassifiziere Texte...")
for i, example in enumerate(dataset):
    text = example["text"]
    label_true = "humorous" if example["humor"] == True else "not humorous"
    
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

# accuracy
correct = sum(r["correct"] for r in results)
print(f"\nAccuracy: {correct}/200 = {correct/200:.2%}")

# false negatives
print("\n--- FALSE NEGATIVES (humorous → predicted as not humorous) ---")
false_negatives = [r for r in results if r["true"] == "humorous" and r["predicted"] == "not humorous"]
for r in false_negatives:
    print(f"  {r['text']}\n")

# false positives
print("\n--- FALSE POSITIVES (not humorous → predicted as humorous) ---")
false_positives = [r for r in results if r["true"] == "not humorous" and r["predicted"] == "humorous"]
for r in false_positives:
    print(f"  {r['text']}\n")

print(f"Total False Negatives: {len(false_negatives)}")
print(f"Total False Positives: {len(false_positives)}")

# generation of confusion matrix
y_true = [r["true"] for r in results]
y_pred = [r["predicted"] for r in results]

cm = confusion_matrix(y_true, y_pred, labels=["humorous", "not humorous"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["humorous", "not humorous"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix – Zero-Shot Humor Recognition")
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
print("Confusion matrix saved as confusion_matrix.png")
