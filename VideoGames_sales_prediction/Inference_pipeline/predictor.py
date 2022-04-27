import model
import pandas as pd

preprocess_data = model.Preprocess('Test.csv')

scaled_x, x = preprocess_data.clean_data()

predicted_df = preprocess_data.predict(x)

predicted_df.to_csv('prediction.csv')