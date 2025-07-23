from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# Create the checkpoints directory in your Drive
checkpoint_dir = '/content/drive/MyDrive/Apple_dataset/CheckPoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# Set path to save model after every epoch
checkpoint_path = os.path.join(checkpoint_dir, 'model_epoch_{epoch:02d}.keras')

# Define checkpoint callback — saves model after every epoch
checkpoint_cb = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=False,  # saves the full model
    verbose=1,
    save_best_only=False      # save after every epoch, not just the best one
)

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[early_stop, checkpoint_cb]  # <-- This avoids unnecessary long training
)