import os
import json
import tensorflow as tf
from tensorflow import keras

def check_and_create_model():
    """检查并创建必要的模型文件"""
    
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    
    # 检查模型文件是否存在
    model_json_path = os.path.join(model_dir, 'model.json')
    weights_path = os.path.join(model_dir, 'weights.bin')
    
    if not os.path.exists(model_json_path):
        print("Creating model files...")
        create_simple_model()
    else:
        print("Model files already exist")
    
    # 验证模型文件
    try:
        with open(model_json_path, 'r') as f:
            model_config = json.load(f)
        print("Model JSON is valid")
        
        # 检查权重文件
        if os.path.exists(weights_path):
            file_size = os.path.getsize(weights_path)
            print(f"Weights file exists, size: {file_size} bytes")
        else:
            print("Weights file missing")
            
    except Exception as e:
        print(f"Error validating model files: {e}")
        create_simple_model()

def create_simple_model():
    """创建一个简单的DNA分类模型"""
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(1000, 4)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(4, activation='softmax', name='output')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 保存为TensorFlow.js格式
    import tensorflowjs as tfjs
    tfjs.converters.save_keras_model(model, 'models/')
    
    print("Simple model created and saved successfully")

if __name__ == "__main__":
    check_and_create_model()
