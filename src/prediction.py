# import mlflow
# mlflow.set_tracking_uri("http://localhost:5001")

# from basic_spam import preprocess_text


# text = "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot!"


# Run_ID='241d005a3872498182883d60d33ffb9d'
# # Loading vector and model
# logged_vect = f'runs:/{Run_ID}/vectorizer'
# logged_model = f'runs:/{Run_ID}/model'

# loaded_vect = mlflow.sklearn.load_model(logged_vect)
# loaded_model = mlflow.pyfunc.load_model(logged_model)


# processed_text = preprocess_text(text)

# vectorized_text = loaded_vect.transform([processed_text])

# prediction = loaded_model.predict(vectorized_text)
# print(f"Predicted label: {prediction[0]}")

# reg_models=mlflow.search_model_versions()
# print(len(reg_models))
#print(reg_models[0])
#from mlflow.tracking import MlflowClient

# # Initialize the MLflow client
#client = MlflowClient()

# # Model name and version
# model_name = "Spamfilter"  # Replace with your model name
# version = 3               # Replace with the version to promote

# # Transition the model to the Production stage
# client.transition_model_version_stage(
#     name=model_name,
#     version=version,
#     stage="None"
# )
#print([[model.name,model.version,model.status, model.current_stage]for model in mlflow.search_model_versions()])

#print(f"Model {model_name} version {version} has been moved to Production.")
