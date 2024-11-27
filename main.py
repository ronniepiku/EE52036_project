import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from tf_keras.regularizers import l2
from tf_keras.layers import Input, Dense, Dropout, BatchNormalization, LSTM, Bidirectional
from tf_keras.models import Sequential

def preprocess_and_predict(input_data, label_mapping):
    # Load the scaler and LDA transformer
    scaler = StandardScaler()
    lda = LDA()

    # Load the feature names
    features = pd.read_csv('UCI HAR Dataset/UCI HAR Dataset/features.txt', sep='\s+', header=None)

    # Create unique feature names by combining the index and the feature name
    unique_column_names = [f"{row[0]}_{row[1]}" for index, row in features.iterrows()]

    # Loading training data to fit scaler
    X_train = pd.read_csv('UCI HAR Dataset/UCI HAR Dataset/train/X_train.txt', sep='\s+', header=None, names=unique_column_names)
    y_train = pd.read_csv('UCI HAR Dataset/UCI HAR Dataset/train/y_train.txt', sep='\s+', header=None)

    # Convert input_data to DataFrame with correct feature names
    input_data_df = pd.DataFrame(input_data, columns=unique_column_names)

    # Fit the scaler on the training data
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    # Transform the input data
    input_data_scaled = scaler.transform(input_data_df)

    # Load the trained model
    print("Model Loading...")

    model = Sequential()
    model.add(Input(shape=(1, 5)))  # Example: 1 timestep, 5 features per timestep
    model.add(Bidirectional(LSTM(16, return_sequences=False, kernel_regularizer=l2(0.0025732713906751817))))
    model.add(BatchNormalization())
    model.add(Dropout(0.27765180873653916))
    model.add(Dense(6, activation='softmax'))

    model.load_weights('./final_model_weights/final_model_weights').expect_partial()

    print("Model Loaded")

    # Apply LDA transformation
    input_data_lda = lda.fit(X_train_scaled, np.ravel(y_train)).transform(input_data_scaled)

    # Reshape input for LSTM (e.g., batch_size, timesteps, features)
    input_data_reshaped = input_data_lda.reshape((input_data_lda.shape[0], 1, input_data_lda.shape[1]))

    # Make predictions
    predictions = model.predict(input_data_reshaped)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Map the predicted class index to the activity label
    predicted_activity = label_mapping[predicted_class]

    return predicted_activity


if __name__ == "__main__":
    label_mapping = {
        1: "Walking",
        2: "Walking Upstairs",
        3: "Walking Downstairs",
        4: "Sitting",
        5: "Standing",
        6: "Laying"
    }

    subject_number = 1
    data_idx = 1

    subject_df = pd.read_csv('UCI HAR Dataset/UCI HAR Dataset/train/subject_train.txt', sep='\s+', header=None)
    subject_idx = subject_df.index[subject_df[0] == subject_number].tolist()


    input_data = pd.read_csv('UCI HAR Dataset/UCI HAR Dataset/train/X_train.txt', sep='\s+', header=None)
    input_row = input_data.iloc[subject_idx[data_idx]]

    activity = preprocess_and_predict(input_row.values.reshape(1, -1), label_mapping)
    print(f"Predicted Activity: {activity}")
