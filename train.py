import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from model import build_cnn
from utils import load_cifar10, preprocess, get_generators
from sklearn.model_selection import train_test_split

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 50
MODEL_DIR = 'save_model'
os.makedirs(MODEL_DIR, exist_ok=True)

# Load and preprocess data
x_train, y_train, x_test, y_test = load_cifar10()
x_train = preprocess(x_train)
x_test = preprocess(x_test)

# Split validation
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

# Generators with data augmentation
train_gen, val_gen = get_generators(x_train, y_train, x_val, y_val, batch_size=BATCH_SIZE, augment=True)

# Build and compile model
model = build_cnn(input_shape=x_train.shape[1:], num_classes=10)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint(os.path.join(MODEL_DIR, 'best_model.keras'), monitor='val_accuracy', save_best_only=True, verbose=1)
early = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Train model
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    steps_per_epoch=len(train_gen),
    validation_steps=len(val_gen),
    callbacks=[checkpoint, early, reduce_lr]
)

# Save final model
model.save(os.path.join(MODEL_DIR, 'final_model.keras'))

# Evaluate on test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")

# Optionally save training history
import pickle
os.makedirs('results', exist_ok=True)
with open(os.path.join('results', 'history.pkl'), 'wb') as f:
    pickle.dump(history.history, f)
