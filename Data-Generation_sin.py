import math
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

x_values_tr_deg = np.random.uniform(1, 361, 700)
x_values_gt_deg = np.random.uniform(1, 361, 300)

x_values_tr_rad = np.radians(x_values_tr_deg)
x_values_gt_rad = np.radians(x_values_gt_deg)

y_values_tr = np.sin(x_values_tr_rad)
y_values_gt = np.sin(x_values_gt_rad)

y_values_tr_error = y_values_tr  + np.random.uniform(-0.3, 0.3, 700)
y_values_gt_error = y_values_gt  + np.random.uniform(-0.3, 0.3, 300)

plt.scatter(x_values_tr_deg, y_values_tr_error, c = "pink")
plt.scatter(x_values_gt_deg, y_values_gt_error, c = "purple")
#plt.show()

df_training_data = pd.DataFrame({
    "x_values": x_values_tr_deg,
    "y_values": y_values_tr_error
})

# In this case i named the gt (ground truth) data validation data
df_validation_data = pd.DataFrame({
    "x_values": x_values_gt_deg,
    "y_values": y_values_gt_error
})

print(df_validation_data)
print(df_training_data)

df_training_data.to_csv("data/training_data.csv", header = False, index = False)
df_validation_data.to_csv("data/validation_data.csv", header = False, index = False)