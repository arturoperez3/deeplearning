import os
import struct
import sqlite3
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# connect to the sql database 
connection = sqlite3.connect("/Users/Arturo1/Desktop/soccer/database.sqlite")

tables = pd.read_sql("""SELECT *
                        FROM sqlite_master
                        WHERE type='table';""", connection)

tables


# overall_rating = labels

# X = [potential, crossing, finishing, heading_accuracy, short_passing, volleys, dribbling, 
# free_kick_accuracy, long_passing, ball_control, acceleration, sprint_speed, agility, reactions
# balance, shot_power, jumping, stamina, strength, long_shots, interceptions, positioning,
# vision, penalties, marking, standing_tackle, sliding_tackle]

data = pd.read_sql("""SELECT overall_rating, potential, crossing, finishing, heading_accuracy, short_passing, volleys, dribbling,
                            free_kick_accuracy, long_passing, ball_control, acceleration, sprint_speed, agility, reactions
                            balance, shot_power, jumping, stamina, strength, long_shots, interceptions, positioning,
                            vision, penalties, marking, standing_tackle, sliding_tackle
                            FROM Player_Attributes
                            """, connection)

data = data.dropna()

data = data.values

print(data.shape)

X = data[:, 1:]
Y = data[:, 1]

print(X.shape)
print(Y.shape)

XTest1 = data[:10000, :]
YTest1 = data[:10000, :]

XTest2 = data[10000:20000, :]
YTest2 = data[10000:20000, :]

XTest3 = data[20000:30000, :]
YTest3 = data[20000:30000, :]

XTrain = data[30000:50000, :]
YTrain = data[30000:50000, :]

regressor = tf.contrib.learn.DNNRegressor(feature_columns=XTrain, 
                                          activation_fn = tf.nn.relu, hidden_units=[200, 100, 50, 25, 12])#,

regressor.fit(XTrain, steps=2000)
ev = regressor.evaluate(Xtrain, steps=1)



