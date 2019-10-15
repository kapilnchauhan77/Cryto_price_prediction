import time
import random
import numpy as np
import pandas as pd 
import tensorflow as tf 
from collections import deque
from sklearn import preprocessing
from tensorflow.keras.models import load_model

files = ['BCH-USD.csv', 'BTC-USD.csv', 'ETH-USD.csv', 'LTC-USD.csv'] 

main_df = pd.DataFrame()

interval = 60
file_to_predict = 'LTC-USD.csv'
future_interval = 3
EPOCHS = 10
BATCH_SIZE = 64
nm = f"Predcting - {file_to_predict}, Sequence length - {interval}, epochs - {EPOCHS}, batch size -{BATCH_SIZE} --Pred--{time.time()}"


def classify(future, current):
	if float(future)>float(current):
		return 1
	else:
		return 0

def prepros(df):
	df = df.drop('future', axis=1)

	for col in df.columns:
		if col != 'target':
			df[col] = df[col].pct_change()
			df.dropna(inplace=True)
			df[col]=preprocessing.scale(df[col].values)

	df.dropna(inplace=True)
	sequential_data = []
	prev_days = deque(maxlen=interval)

	for i in df.values:
		prev_days.append([n for n in i[:-1]])
		if len(prev_days) == interval:
			sequential_data.append([np.array(prev_days), i[-1]])
			
	random.shuffle(sequential_data)	

	buys = []
	sells = []
	
	for seq, target in sequential_data:
		if target == 0:
			sells.append([seq, target])
		else:
			buys.append([seq, target])

	lower = min(len(buys), len(sells))

	buys = buys[:lower]
	sells = sells[:lower]		

	random.shuffle(buys)
	random.shuffle(sells)

	sequential_data = buys + sells
	random.shuffle(sequential_data)

	x = []
	y = []

	for seq, target in sequential_data:
		x.append(seq)
		y.append(target)

	return x, y


for file in files:

	df = pd.read_csv(f'crypto_data/{file}', names=['Time', 'low', 'high', 'open', 'close', 'volume'])

	df.rename(columns={'close' : f'close_{file}', 'volume' : f'volume_{file}'}, inplace=True)

	df.set_index('Time', inplace=True)
	df = df[[f'close_{file}', f'volume_{file}']]

	if len(main_df) == 0:
		main_df = df

	else:
		main_df = main_df.join(df)

main_df['future'] = main_df[f'close_{file_to_predict}'].shift(-future_interval)

main_df['target'] = list(map(classify, main_df['future'], main_df[f'close_{file_to_predict}']))

main_df = main_df.fillna(method='ffill')
main_df = main_df.dropna()

times = sorted(main_df.index.values)
last_5pc = times[-int(len(times)*0.05)]

validation_df = main_df[(main_df.index>=last_5pc)]
main_df = main_df[(main_df.index<last_5pc)]

x_train, y_train = prepros(main_df)
x_test, y_test = prepros(validation_df)

print(f"Train data: {len(x_train)}, Validation data : {len(x_test)}")
print(f"Don't buys: {y_train.count(0)}, Buys: {y_train.count(1)}")
print(f"Don't buys in validation: {y_test.count(0)}, Buys: {y_test.count(1)}")

x_train=np.array(x_train)
x_test=np.array(x_test)

model = load_model('Model.h5')
print(model.predict(x_test)[1])
print('Real:')
print(y_test[1])
input()
