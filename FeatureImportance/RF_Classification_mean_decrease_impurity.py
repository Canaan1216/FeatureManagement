from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor

# Load boston housing dataset as an example

boston = load_boston()
# print boston["feature_names"]
print boston["DESCR"]

X = boston["data"]
Y = boston["target"]
names = boston["feature_names"]
rf = RandomForestRegressor()
rf.fit(X, Y)
print "Features sorted by their score:"
print sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names),
             reverse=True)