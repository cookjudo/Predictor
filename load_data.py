import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

raw_data = pd.read_csv("housing.csv")
raw_data = raw_data.dropna()

data = pd.get_dummies(raw_data, columns=["ocean_proximity"], dtype=int)

##print(raw_data.head())

shuffled_data = data.sample(n=len(data), random_state=1)

##print(shuffled_data.head())

X = shuffled_data.drop("median_house_value", axis=1)
y = shuffled_data["median_house_value"].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train)

poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

X_train_data = X_train_poly
X_test_data = X_test_poly

y_scaler = StandardScaler()

y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

X_train_tensor = torch.from_numpy(X_train_data).float()
y_train_tensor = torch.from_numpy(y_train_scaled).float()
X_test_tensor = torch.from_numpy(X_test_data).float()
y_test_tensor = torch.from_numpy(y_test_scaled).float()

input_size = X_train_data.shape[1]

class neural(nn.Module):
    def __init__(self, input_size):
        super(neural, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU() 
        self.fc2 = nn.Linear(64, 1) 

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
model = neural(input_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)

y_test_scaled_np = y_test_tensor.numpy()
y_pred_scaled_np = y_pred.numpy()
y_test_final = y_scaler.inverse_transform(y_test_scaled_np)
y_pred_final = y_scaler.inverse_transform(y_pred_scaled_np)
mse = mean_squared_error(y_test_final, y_pred_final)
r2 = r2_score(y_test_final, y_pred_final)
rmse = np.sqrt(mse)



W1 = model.fc1.weight.data.numpy()
b1 = model.fc1.bias.data.numpy()

W2 = model.fc2.weight.data.numpy()
b2 = model.fc2.bias.data.numpy()

plt.figure(figsize=(10, 6))
plt.scatter(y_test_final, y_pred_final, alpha=0.6, color='darkorange', label='Predicted vs. Actual Points')

min_val = min(y_test_final.min(), y_pred_final.min())
max_val = max(y_test_final.max(), y_pred_final.max())
perfect_line = np.linspace(min_val, max_val, 100)

plt.plot(perfect_line, perfect_line, color='blue', linestyle='--', linewidth=2, label='Perfect Prediction Line (y=x)')

plt.xlabel('Actual Median House Value ($)')
plt.ylabel('Predicted Median House Value ($)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.show()
