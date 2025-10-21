import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json

class DNASequenceClassifier:
    def __init__(self, sequence_length=1000):
        self.sequence_length = sequence_length
        self.model = None
        self.label_encoder = LabelEncoder()
        self.class_names = ['Human', 'Bacteria', 'Virus', 'Plant']
        
    def one_hot_encode_sequence(self, sequence):
        """Convert DNA sequence to one-hot encoding"""
        mapping = {'A': [1, 0, 0, 0], 
                  'T': [0, 1, 0, 0],
                  'C': [0, 0, 1, 0],
                  'G': [0, 0, 0, 1]}
        
        # Convert to uppercase and remove invalid characters
        sequence = ''.join([base for base in sequence.upper() if base in 'ATCG'])
        
        # Pad or truncate to fixed length
        if len(sequence) < self.sequence_length:
            sequence += 'N' * (self.sequence_length - len(sequence))
        else:
            sequence = sequence[:self.sequence_length]
        
        # One-hot encoding
        encoded = []
        for base in sequence:
            if base in mapping:
                encoded.append(mapping[base])
            else:
                encoded.append([0, 0, 0, 0])  # For 'N' or other characters
        
        return np.array(encoded)
    
    def load_and_preprocess_data(self, csv_file_path):
        """Load and preprocess the DNA sequence dataset"""
        print("Loading dataset...")
        
        # Read the Excel file (since your file appears to be Excel format)
        df = pd.read_excel(csv_file_path, engine='openpyxl')
        
        # Assuming the dataset has 'sequence' and 'label' columns
        # Adjust column names based on your actual dataset structure
        sequences = df['sequence'].values  # Replace 'sequence' with actual column name
        labels = df['label'].values        # Replace 'label' with actual column name
        
        print(f"Loaded {len(sequences)} sequences")
        
        # Preprocess sequences
        X = np.array([self.one_hot_encode_sequence(seq) for seq in sequences])
        
        # Encode labels
        y = self.label_encoder.fit_transform(labels)
        
        return X, y
    
    def build_model(self, input_shape):
        """Build the neural network model"""
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=input_shape),
            
            # Convolutional layers for sequence pattern recognition
            layers.Conv1D(64, kernel_size=10, activation='relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.BatchNormalization(),
            
            layers.Conv1D(128, kernel_size=8, activation='relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.BatchNormalization(),
            
            layers.Conv1D(256, kernel_size=6, activation='relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.BatchNormalization(),
            
            layers.Conv1D(512, kernel_size=4, activation='relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.BatchNormalization(),
            
            # Global pooling and dense layers
            layers.GlobalAveragePooling1D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.BatchNormalization(),
            
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(4, activation='softmax')  # 4 classes
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X, y, epochs=50, batch_size=32):
        """Train the model"""
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Build model
        self.model = self.build_model(X_train[0].shape)
        
        print("Model architecture:")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
            keras.callbacks.ModelCheckpoint(
                'best_model.h5', save_best_only=True, monitor='val_accuracy'
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate(self, X, y):
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        loss, accuracy = self.model.evaluate(X, y, verbose=0)
        return loss, accuracy
    
    def predict(self, sequence):
        """Predict class for a single sequence"""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        encoded_seq = self.one_hot_encode_sequence(sequence)
        encoded_seq = np.expand_dims(encoded_seq, axis=0)
        
        predictions = self.model.predict(encoded_seq)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        return {
            'class': self.label_encoder.inverse_transform([predicted_class])[0],
            'confidence': float(confidence),
            'all_probabilities': {
                self.class_names[i]: float(prob) 
                for i, prob in enumerate(predictions[0])
            }
        }
    
    def save_model(self, model_path='models/dna_classifier'):
        """Save model in TensorFlow.js format"""
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save TensorFlow model
        self.model.save('dna_classifier.h5')
        
        # Convert to TensorFlow.js format
        import tensorflowjs as tfjs
        tfjs.converters.save_keras_model(self.model, model_path)
        
        # Save label encoder information
        model_info = {
            'class_names': self.class_names,
            'sequence_length': self.sequence_length,
            'label_encoder_classes': self.label_encoder.classes_.tolist()
        }
        
        with open(f'{model_path}/model_info.json', 'w') as f:
            json.dump(model_info, f)
        
        print(f"Model saved to {model_path}")

def main():
    # Initialize classifier
    classifier = DNASequenceClassifier(sequence_length=1000)
    
    try:
        # Load and preprocess data
        X, y = classifier.load_and_preprocess_data('data/synthetic_dna_dataset_test.csv')
        
        print(f"Data shape: {X.shape}")
        print(f"Number of classes: {len(np.unique(y))}")
        
        # Train model
        print("Starting model training...")
        history = classifier.train(X, y, epochs=50, batch_size=32)
        
        # Evaluate model
        loss, accuracy = classifier.evaluate(X, y)
        print(f"Final Model Accuracy: {accuracy:.4f}")
        print(f"Final Model Loss: {loss:.4f}")
        
        # Save model
        classifier.save_model('models/')
        
        print("Model training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        # Create a simple demo model if training fails
        create_demo_model()

def create_demo_model():
    """Create a simple demo model for testing"""
    print("Creating demo model...")
    
    # Generate synthetic data for demo
    np.random.seed(42)
    
    # Create a simple model
    model = keras.Sequential([
        layers.Input(shape=(1000, 4)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(4, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Save demo model
    import tensorflowjs as tfjs
    tfjs.converters.save_keras_model(model, 'models/')
    
    # Save model info
    model_info = {
        'class_names': ['Human', 'Bacteria', 'Virus', 'Plant'],
        'sequence_length': 1000,
        'label_encoder_classes': ['Human', 'Bacteria', 'Virus', 'Plant']
    }
    
    with open('models/model_info.json', 'w') as f:
        json.dump(model_info, f)
    
    print("Demo model created!")

if __name__ == "__main__":
    main()
