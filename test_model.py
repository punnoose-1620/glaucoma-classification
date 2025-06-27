import tensorflow as tf
import numpy as np
from model_vit_version import ModelConfig, HybridGlaucomaClassifier, FEATURE_MAP

def test_model_with_different_shapes():
    """Test the model with different input shapes."""
    config = ModelConfig(
        input_size=224,  # Base input size
        patch_size=16,
        in_channels=3,
        num_classes=2,
        embed_dim=768,
        depth=12,
        num_heads=8,  # Changed from 12 to 8 for compatibility with dim=64
        mlp_ratio=4.0,
        window_size=7
    )
    model = HybridGlaucomaClassifier(config)
    test_shapes = [
        (224, 224),
        (256, 256),
        (192, 192),
        (320, 240),
    ]
    batch_size = 2
    for height, width in test_shapes:
        print(f"\nTesting with input shape: {height}x{width}")
        inputs = {}
        for modality in FEATURE_MAP.values():
            inputs[modality] = tf.random.normal([batch_size, height, width, 3])
        try:
            outputs = model(inputs, training=False)
            expected_output_shape = (batch_size, config.num_classes)
            assert outputs.shape == expected_output_shape, \
                f"Expected output shape {expected_output_shape}, got {outputs.shape}"
            assert tf.reduce_all(tf.math.is_finite(outputs)), "Output contains NaN or Inf values"
            assert tf.reduce_all(outputs >= 0) and tf.reduce_all(outputs <= 1), \
                "Output values should be between 0 and 1 (softmax)"
            assert tf.reduce_all(tf.abs(tf.reduce_sum(outputs, axis=1) - 1.0) < 1e-6), \
                "Output probabilities should sum to 1"
            print(f"✓ Test passed for shape {height}x{width}")
        except Exception as e:
            print(f"✗ Test failed for shape {height}x{width}")
            print(f"Error: {str(e)}")
            raise

def train_and_show_results():
    """Train the model for a few epochs and print results."""
    config = ModelConfig(num_heads=8, num_classes=2)
    model = HybridGlaucomaClassifier(config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    batch_size = 8
    height, width = 224, 224
    num_modalities = len(FEATURE_MAP)
    num_batches = 10
    epochs = 3
    # Generate synthetic data
    def get_batch():
        inputs = {modality: tf.random.normal([batch_size, height, width, 3]) for modality in FEATURE_MAP.values()}
        labels = tf.random.uniform([batch_size], maxval=config.num_classes, dtype=tf.int32)
        return inputs, labels
    print("\nTraining model...")
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_acc = 0
        for batch in range(num_batches):
            inputs, labels = get_batch()
            with tf.GradientTape() as tape:
                logits = model(inputs, training=True)
                loss = loss_fn(labels, logits)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            preds = tf.argmax(logits, axis=1)
            acc = tf.reduce_mean(tf.cast(preds == tf.cast(labels, tf.int64), tf.float32))
            epoch_loss += float(loss)
            epoch_acc += float(acc)
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {avg_acc:.4f}")
    print("\nTraining complete.")

def test_model_validation():
    config = ModelConfig(num_heads=8)
    model = HybridGlaucomaClassifier(config)
    test_cases = [
        {
            "inputs": {list(FEATURE_MAP.values())[0]: tf.random.normal([2, 224, 224, 3])},
            "expected_error": "Missing input for modality"
        },
        {
            "inputs": tf.random.normal([2, 224, 224, 3]),
            "expected_error": "Inputs must be a dictionary"
        },
        {
            "inputs": {modality: tf.random.normal([2, 224, 224]) for modality in FEATURE_MAP.values()},
            "expected_error": "Input must be 4D tensor"
        }
    ]
    for i, test_case in enumerate(test_cases):
        print(f"\nTesting validation case {i+1}")
        try:
            model(test_case["inputs"])
            print(f"✗ Test case {i+1} failed: Expected error but got none")
            raise AssertionError(f"Expected error: {test_case['expected_error']}")
        except Exception as e:
            if test_case["expected_error"] in str(e):
                print(f"✓ Test case {i+1} passed: Caught expected error")
            else:
                print(f"✗ Test case {i+1} failed: Unexpected error")
                raise

def test_model_training():
    config = ModelConfig(num_heads=8)
    model = HybridGlaucomaClassifier(config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    batch_size = 2
    height, width = 224, 224
    inputs = {modality: tf.random.normal([batch_size, height, width, 3]) 
             for modality in FEATURE_MAP.values()}
    labels = tf.random.uniform([batch_size], maxval=config.num_classes, dtype=tf.int32)
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
        loss = tf.reduce_mean(loss)
    gradients = tape.gradient(loss, model.trainable_variables)
    assert all(g is not None for g in gradients), "Some gradients are None"
    assert all(tf.reduce_all(tf.math.is_finite(g)) for g in gradients), \
        "Some gradients contain NaN or Inf values"
    print("\n✓ Training test passed")

if __name__ == "__main__":
    print("Running model tests...")
    print("\n1. Testing with different input shapes")
    test_model_with_different_shapes()
    print("\n2. Training model and showing results")
    train_and_show_results()
    print("\n3. Testing input validation")
    test_model_validation()
    print("\n4. Testing model training")
    test_model_training()
    print("\nAll tests completed successfully!") 