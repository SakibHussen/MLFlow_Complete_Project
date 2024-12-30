
from sklearn.model_selection import train_test_split
import Data_ingestion
from basic_spam import preprocess_text,train_model,eval_met
import pandas as pd


# Load the processed dataset dynamically from data_ingestion.saved_file_path
df = pd.read_csv("/Users/sakibhussen/Desktop/Youtube_ML_OP_Series/MLFlow_Complete_Project/datasets/SPAM_DATASETS.csv")

# Apply text preprocessing on the message column
df['processed_message'] = df.message.apply(preprocess_text)

# Split the data into features (x) and labels (y)
x = df['processed_message']
y = df['label']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the model
vectorizer, model , train_acc = train_model(x_train, y_train, n=10, c='entropy', d=2)

# Evaluate the model
y_pred = model.predict(vectorizer.transform(x_test))
acc, prc, rec, f1 = eval_met(y_pred, y_test)

# Print the Results
print(f"Training Accuracy: {train_acc*100:.3f} %")
print(f"Validation Accuracy: {acc*100:.3f} %")
print(f"Precision Score: {prc*100:.3f} %")
print(f"Recall Score: {rec*100:.3f} %")
print(f"F1 Score: {f1*100:.3f} %")
