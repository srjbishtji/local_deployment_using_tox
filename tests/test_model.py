from ml_model.train import train_model
from ml_model.predict import predict

def test_train_model():
    # Train the model
    model = train_model()
    assert model is not None  # Ensure the model is trained

def test_prediction():
    # Test if prediction works with the model
    # Update the input to match the number of features (8 for California housing dataset)
    sample_input = [1.0, 0.0, 37.0, 2.0, 0.0, 2.0, 4.0, 3.0]  # Example with 8 features
    prediction = predict(sample_input)
    assert len(prediction) == 1  # Ensure prediction is returned
