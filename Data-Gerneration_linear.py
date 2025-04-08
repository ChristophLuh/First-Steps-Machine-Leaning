import math
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

x_values_tr = np.array(random.sample(range(1, 100), 70))
x_values_gt = np.array(random.sample(range(1, 100), 30))

y_values_tr_error = 2.3 * x_values_tr + 4  + np.random.uniform(-30, 30, 70)
y_values_gt_error = 2.3 * x_values_gt + 4  + np.random.uniform(-30, 30, 30)

plt.scatter(x_values_tr, y_values_tr_error, c = "pink")
plt.scatter(x_values_gt, y_values_gt_error, c = "purple")
plt.show()

df_training_data = pd.DataFrame({
    "x_values": x_values_tr,
    "y_values": y_values_tr_error
})

# In this case i named the gt (ground truth) data validation data
df_validation_data = pd.DataFrame({
    "x_values": x_values_gt,
    "y_values": y_values_gt_error
})

print(df_validation_data)
print(df_training_data)

df_training_data.to_csv("data/training_data_linear.csv", header = False, index = False)
df_validation_data.to_csv("data/validation_data_linear.csv", header = False, index = False)