import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

FILE_PATH = 'housing.csv'
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.1
INPUT_DIM = 13
HIDDEN_DIM = 64

try:
    df = pd.read_csv(FILE_PATH)
    
    # 1.
    df = df.dropna()
    
    # 2. 
    if 'ocean_proximity' in df.columns:
        df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=False)

    TARGET_COL = 'median_house_value'
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    # 3.
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)

    # 4.
    tensor_x = torch.tensor(X_normalized, dtype=torch.float32)
    tensor_y = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        tensor_x, tensor_y, test_size=0.2, random_state=42
    )

    # 5.
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    INPUT_DIM = X_train.shape[1]

except FileNotFoundError:
    print(f"Error: The file '{FILE_PATH}' was not found.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during data processing: {e}")
    exit()

class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = RegressionModel(INPUT_DIM, HIDDEN_DIM)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_loss_history = []
for epoch in range(EPOCHS):
    model.train() 
    total_loss = 0
    
    for batch_features, batch_targets in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_features)
        loss = criterion(predictions, batch_targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_features.size(0)

    avg_loss = total_loss / len(train_dataset)
    train_loss_history.append(avg_loss)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}')

model.eval() 
test_predictions = []
true_targets = []

with torch.no_grad():
    for features, targets in test_loader:
        predictions = model(features)
        test_predictions.extend(predictions.squeeze().tolist())
        true_targets.extend(targets.squeeze().tolist())

mse = np.mean((np.array(test_predictions) - np.array(true_targets)) ** 2)
rmse = np.sqrt(mse)

print(f"Test Set RMSE: {rmse:.2f}")



plt.figure(figsize=(10, 6))
plt.scatter(true_targets, test_predictions, alpha=0.3)
plt.plot([min(true_targets), max(true_targets)], 
         [min(true_targets), max(true_targets)], 
         color='red', linestyle='--', linewidth=2, label='Perfect Prediction') 
plt.xlabel("True Median House Value")
plt.ylabel("Predicted Median House Value")
plt.title(f"True vs. Predicted House Values (RMSE: {rmse:.2f})")
plt.grid(True)
plt.savefig('house_price_prediction_plot.png')
print("Plot saved to 'house_price_prediction_plot.png'")