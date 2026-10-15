import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

df = pd.read_csv('mcs_ds_edited_iter_shuffled.csv')
X = df[['anchor_ratio', 'trans_range', 'node_density', 'iterations']]
y = df['ale']

# 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# standardize (fit on train only)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    'Linear': LinearRegression(),
    'SVM': SVR(),
    'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
    'Neural Network': MLPRegressor(max_iter=2000, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    residuals = np.asarray(y_test).ravel() - preds

    mae = mean_absolute_error(y_test, preds)
    stat, p_value = stats.normaltest(residuals)

    print(f"Model: {name}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Normality p-value: {p_value:.6g}\n")

    fig, ax = plt.subplots(figsize=(6, 4))
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title(f"{name}: normal Q-Q plot")
    plt.tight_layout()
    plt.show()