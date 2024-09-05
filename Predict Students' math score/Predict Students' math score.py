import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from lazypredict.Supervised import LazyRegressor

data = pd.read_csv("StudentScore.xls")
# Take a first overall look about data
# profile = ProfileReport(data)
# profile.to_file("StudentScore.html")

# Determine features and label
features = data.drop(["math score"], axis=1)
target = data["math score"]


# Split data

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=23)

# Data Preprocessing
# Show out those values in each categorical features
education_values = ['some high school', 'high school', 'some college', "associate's degree", "bachelor's degree",
                    "master's degree"]
gender_values = ["male", "female"]
lunch_values = x_train["lunch"].unique()
test_values = x_train["test preparation course"].unique()
# Making Pipeline for each data types
num_transform = Pipeline([('imp', SimpleImputer(strategy='median')),
                          ('scaler', StandardScaler())])
ord_transform = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                          ('scaler', OrdinalEncoder(categories=[education_values, gender_values, lunch_values, test_values]))])
nom_transform = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                          ('scaler', OneHotEncoder(sparse_output=True))])
# Making preprocessor for the whole features
preprocessor = ColumnTransformer([('num', num_transform, ['reading score', "writing score"]),
                                  ('ord', ord_transform, ["parental level of education", "gender", "lunch", "test preparation course"]),
                                  ('nom', nom_transform, ["race/ethnicity"])])
# Model fit

model = Pipeline([('preprocessor', preprocessor), ("model", LinearRegression())])
model1 = Pipeline([('preprocessor', preprocessor), ("model", RandomForestRegressor())])
params = {
    "preprocessor__num__imp__strategy": ["mean", "median", "most_frequent"],
    "model__n_estimators": [100, 200, 300],
    "model__criterion": ["squared_error", "absolute_error", "friedman_mse", "poisson"]
}
# model.fit(x_train, y_train)
# y_predict = model.predict(x_train)
# for i, j in zip(y_train, y_predict):
#     print("Prediction: {} Actual: {}".format(i, j))
grid_search = GridSearchCV(estimator=model1, param_grid=params, scoring="r2", verbose=1,cv=5, n_jobs=-1)
grid_search.fit(x_train, y_train)
print(grid_search.best_score_)
print(grid_search.best_params_)
y_predict = grid_search.predict(x_test)
print("MAE: {}".format(mean_absolute_error(y_true=y_test, y_pred=y_predict)))
print("MSE: {}".format(mean_squared_error(y_test, y_predict)))
print("R2: {}".format(r2_score(y_test, y_predict)))
reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = reg.fit(x_train, x_test, y_train, y_test)