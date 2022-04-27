import pandas as pd
from sklearn.preprocessing  import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

file_name = 'imputer.pkl'
xgb_impute = pickle.load(open(file_name, "rb"))
model = pickle.load(open('sgd_model.pkl', 'rb'))
le = pickle.load(open('encoder.pkl', 'rb'))
scaler = pickle.load(open('scaler_std.pkl','rb'))

class Preprocess:
	def __init__(self, data):
		self.df = pd.read_csv(data)
		self.df_pred = self.df['Name']

	def clean_data(self):
		data = self.df
		df_clean = data.drop(['Year_of_Release', 'Publisher', 'Developer', 'Name', 'Critic_Score', 'Critic_Count', 'User_Score','User_Count'], axis=1)
		df_rating_test = df_clean['Rating']
		df_dummy_test = df_clean.drop('Rating', axis=1)
		df_dummy_test = pd.get_dummies(df_dummy_test, drop_first=True)
		df_impute_test = pd.concat([df_dummy_test, df_rating_test], axis=1)
		df_impute_test.sort_index(axis=1, inplace=True)
		null_rating = df_impute_test[df_impute_test['Rating'].isnull() == True]
		null_rating = null_rating.drop('Rating', axis=1)
		print(null_rating.shape)
		rating_predicted = xgb_impute.predict(null_rating)
		rating_predicted = le.inverse_transform(rating_predicted)
		null_rating['Rating'] = rating_predicted
		non_null_rating = df_impute_test[df_impute_test['Rating'].isnull() == False]
		final_test_df = pd.concat([null_rating, non_null_rating], axis=0)
		final_test_df.sort_index(inplace=True, ascending=True)
		final_test_df = pd.get_dummies(final_test_df, drop_first=True)
		scaled_test_df = scaler.fit_transform(final_test_df)

		return scaled_test_df, final_test_df


	def predict(self, x):
		prediction = model.predict(x)
		df_pred = pd.DataFrame(self.df_pred)
		df_pred['Global_Sales'] = prediction 
		return df_pred

