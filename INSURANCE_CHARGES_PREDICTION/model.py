import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import pickle

file = pd.read_csv("cleanedfile.csv")
print(file)


x = file.drop(['Z-Score_charges','Z-Score_children', 'Z-Score_bmi', 'Z-Score_age','charges','region_northeast', 'region_northwest', 'region_southeast','region_southwest'], axis=1)

y = file[['charges']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


model = DecisionTreeRegressor(random_state=42)

mod = model.fit(x_train, y_train)


# y_pred= model.predict(x_test)

pickle.dump(mod, open("mod.pkl", "wb"))


