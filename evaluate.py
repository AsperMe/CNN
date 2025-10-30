import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from utils import load_cifar10, preprocess


# Configurations
MODEL_PATH = 'save_model/best_model.keras'
HISTORY_PATH = 'results/history.pkl'
RESULTS_DIR = 'results'

os.makedirs(RESULTS_DIR, exist_ok=True)


# Load History and Model
print("Loading training history and model...")
history = pickle.load(open(HISTORY_PATH, 'rb'))
model = load_model(MODEL_PATH)


# Plot Accuracy
plt.figure(figsize=(8, 6))
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, 'accuracy.png'))
plt.close()


# Plot Loss
plt.figure(figsize=(8, 6))
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, 'loss.png'))
plt.close()

print("Accuracy and Loss plots saved in 'results/' folder.")

# Evaluate on Test Set
print("Loading and preprocessing CIFAR-10 test data...")
x_train, y_train, x_test, y_test = load_cifar10()
x_test = preprocess(x_test)

print("Evaluating model on test set...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")


# Classification Report
print("Generating predictions and classification report...")
preds = model.predict(x_test)
pred_labels = np.argmax(preds, axis=1)
true_labels = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(true_labels, pred_labels, digits=4))


# Confusion Matrix
print("[Creating confusion matrix...")
cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=False, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'))
plt.close()

print("Confusion matrix saved successfully.")
print("Evaluation complete")
