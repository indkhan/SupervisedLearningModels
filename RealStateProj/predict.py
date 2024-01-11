from joblib import load
import numpy as np
clf = load('housingregression.joblib')
price = clf.predict(np.array(
    [[0.7842, 0, 8.14, 0, 0.538, 5.99, 81.7, 4.2579, 4, 307, 21, 386.75, 14.67, 1]]))
print(price)
