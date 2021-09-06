# Positional-Signal-Processing

## Description of the project
Labeled data were collected from smartwatches' sensors. There are three types of sensors included, accelerometer, gyroscope and heart rate sensor. The frequency of accelerometer and gyroscope is 10Hz and the heart rate sensor's is 1Hz. We modeled the data of the sensors as multivariate timeseries and each reading is considered as a random variable. Example array((sample 0), (sample 1)...,(sample n)), and sample i = (random varable 1, random varable 2, ..., random variable k) where n is the total number of samples and k is the total namber of sensors readings in each sample. Every multivariate timeseries is labeled, based on user's activity. The main goal of the project is to create a machine learning method in order to classify each activity according to the sensors' readings.

## Example of the data structure
X -> list(x1, x2, ..., xn), where xi is timeseries.  
y -> list(y1, y2, ..., yn), where yi is the activity id of timeseries xi.  
xi -> np.array((sample 0), (sample 1), ..., (sample k)), where sample i is an array of sensors reading (Acel_X, Acel_Y, Acel_Z, Gyro_X, Gyro_Y, Gyro_Z) and k+1 is the total number of samples for the activity i.  

## Libraries
1. [Numpy](https://numpy.org/): The fundamental package for scientific computing with Python programming language.
2. [Pandas](https://pandas.pydata.org/): Data analysis and manipulation tool built on top of the Python programming language.
3. [Sklearn](https://scikit-learn.org/stable): Tools for predictive data analysis, built on NumPy, SciPy, and Matplotlib.
4. [Seglearn](https://dmbee.github.io/seglearn/): Flexible approach to multivariate time series and contextual data for classification, regression, and forecasting problems.
5. [Matplotlib](https://matplotlib.org/): Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.


