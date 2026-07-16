import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_loader import load_data
from src.preprocessing import encode_categorical, scale_features
from src.model_training import select_features, train_linear_regression, train_polynomial_regression
from src.utils import save_object

# Step 1: Load Data
df = load_data("student_scores.csv")  # Change file name to your dataset

# Step 2: Define features and target
X = df.drop(columns=["Exam_Score"])  # Target column name
y = df["Exam_Score"]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Preprocessing
cat_cols = ["Gender", "Parental_Education"]  # change to your categorical features
scal_cols = ["Attendance", "Hours_Studied", "Sleep_Hours",
             "Previous_Scores", "Tutoring_Sessions", "Physical_Activity"]

x_train, x_test = encode_categorical(x_train, x_test, cat_cols)
x_train, x_test = scale_features(x_train, x_test, scal_cols)

# Step 4: Feature Selection
selected_cols, feat_importance_df = select_features(x_train, y_train)
print("\nSelected Features:", selected_cols)

x_train = x_train[selected_cols]
x_test = x_test[selected_cols]

# Step 5: Train Models
lin_model, lin_mse = train_linear_regression(x_train, y_train, x_test, y_test)
save_object(lin_model, "linear_model.pkl")

poly_model, poly_transformer, poly_mse = train_polynomial_regression(x_train, y_train, x_test, y_test)
