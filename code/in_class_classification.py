# %%
from sklearn.tree import DecisionTreeClassifier, plot_tree
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer

# %%
# Load the breast cancer dataset
cancer = load_breast_cancer(as_frame=True)
X = cancer.data
y = cancer.target
print(cancer.DESCR)

# %%
y_label = [{0: "malignant", 1: "benign"}[i] for i in y]
sns.scatterplot(x=X["mean radius"],
                y=X["mean smoothness"],
                hue=y_label,
                palette="Set1")
# %%
feature_1 = "mean radius"
feature_2 = "mean smoothness"
X = X[[feature_1, feature_2]].values

# %%
# Logistic regression

# BUILD A MODEL: 
model = LogisticRegression(penalty=None).fit(X, y)

# PREDICT AND EVALUATE: 
model.predict_proba(X)
print(model.score(X, y))

# %% Plotting decision boundary

# Create meshgrid
x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

# Compute decision function over the grid
Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
plt.contourf(xx, yy, Z, levels=50, cmap="RdBu", alpha=0.6)  # background
plt.contour(xx, yy, Z, levels=[0], colors='black',
            linewidths=2)  # decision boundary
sns.scatterplot(x=X[:, 0],
                y=X[:, 1],
                hue=y_label,
                edgecolors='k',
                palette="Set1",
                alpha=0.8)
plt.legend()
plt.xlabel(feature_1)
plt.ylabel(feature_2)
plt.title("Logistic Regression Decision Boundary")
plt.show()

# %% DECISION TREE CLASSIFIER
# BUILD A MODEL: 
dt_model = DecisionTreeClassifier(max_depth=3).fit(X, y)
# PREDICT AND EVALUATE: 
print(dt_model.score(X, y))
# %% PLOT DECISION TREE
plot_tree(dt_model, feature_names=[
          feature_1, feature_2], class_names=cancer.target_names, filled=True)
# %% TRY TO BUILD A BETTER CLASSIFIER BY PICKING BETTER FEATURES!
# to do this, you can loop over all pairs of features in the dataset
