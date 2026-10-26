import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("cars.csv")
features = ["year", "km_driven", "fuel", "transmission", "owner_count"]
X, y = df[features], df["selling_price"]
prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"),
                           ["fuel", "transmission"])], remainder="passthrough")
model = make_pipeline(prep, RandomForestRegressor(random_state=42))
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(Xtr, ytr)
print("MAE:", round(mean_absolute_error(yte, model.predict(Xte)), 2))
sample = pd.DataFrame([{"year": int(input("Year: ")), "km_driven": float(input("KM driven: ")),
 "fuel": input("Fuel (Petrol/Diesel/CNG): "),
 "transmission": input("Transmission (Manual/Automatic): "),
 "owner_count": int(input("Owner count: "))}])
print("Estimated car price:", round(model.predict(sample)[0], 2))
