import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import numpy

from prep_data import get_prepared_data, get_all_titles, get_real_numbers,unscale_predicted_vals

# import model from model.py
from model import create_model

# import training function from train.py
from train import train_model

# you can call this function to test a pre-trained model (might be useful while testing)
def test_saved_model(model_path="saved_weights/test_model.pth"):
    # Load model
    model = torch.load(model_path)
    model.eval()

    # Load data
    features, target = get_prepared_data()
    actual_vals= get_real_numbers()
    # Define loss function
    criterion = torch.nn.MSELoss()

    # get list of movie titles
    titles = get_all_titles()

    # Predict across all data
    with torch.no_grad():
        output = model(features)
        predicted_vals= unscale_predicted_vals(output)
        loss = criterion(output, target)
        print(f"\nTest Loss: {loss.item()}")
        # test accuracy
        print(f"Test Accuracy: {1 - loss.item() / target.var()}")

        # print first 10 predictions against actual values
        # For <movie title>, model predicted <prediction>, actual <actual>
        print("\nSample Predictions vs Actual:")
        for i in range(10):
            print(f"For {titles[int(i)]}, model predicted {predicted_vals[i].item()} vs. actual {actual_vals[i].item()}")

        # print best prediction and worst prediction
        # For <movie title>, model predicted <prediction>, actual <actual>
        print("\nBest and Worst Predictions:")
        errors = torch.abs(output - target)
        worst = torch.argmax(errors)
        best = torch.argmin(errors)
        # convert to numpy arrays
        titles = titles.to_numpy()
        output = output.numpy().flatten()
        target = target.numpy().flatten()
        predicted_vals=predicted_vals.flatten()
        actual_vals=actual_vals.to_numpy().flatten()

        print(f"Best Prediction: For {titles[best]}, model predicted {predicted_vals[best].item()} vs. actual {actual_vals[best].item()}")
        print(f"Worst Prediction: For {titles[worst]}, model predicted {predicted_vals[worst].item()} vs. actual {actual_vals[worst].item()}")

# same function but using a new model
def test_new_model():
    # Load data
    features, target = get_prepared_data()

    # create training and validation sets
    # use 80% of the data for training and 20% for validation
    X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2)

    # Define model (feed-forward, two hidden layers)
    model, optimizer = create_model(X_train)

    # Define loss function and optimizer
    criterion = nn.MSELoss()

    # train model
    model = train_model(model, optimizer, criterion, X_train, y_train, X_val, y_val)

    # basic evaluation (more in test.py)
    # Save model
    torch.save(model, "saved_weights/test_model.pth")
    print("Model saved as test_model.pth")

    # Test model
    test_saved_model()

if __name__ == '__main__':
    test_new_model()