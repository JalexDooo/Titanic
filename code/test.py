import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

path = '/home/jonty/python3-test/Titanic/'
train_data = pd.read_csv(path+'train.csv')

def NoAges():
	"""Adding No Ages!!!!"""

	print('-----------Started add NoAges!-----------')

	age = train_data[['Age', 'Survived', 'Fare', 'Parch', 'SibSp', 'Pclass']]
	age_notnull = age.loc[(train_data.Age.notnull())]
	age_isnull = age.loc[(train_data.Age.isnull())]
	# print(age_notnull)
	# print(age_isnull)
	X = age_notnull.values[:, 1:]
	Y = age_notnull.values[:, 0]
	rfr = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
	rfr.fit(X, Y)
	predictAges = rfr.predict(age_isnull.values[:, 1:])
	# print('++++++++++++++++++')
	# print(predictAges)
	# print('++++++++++++++++++')
	train_data.loc[(train_data.Age.isnull()), 'Age'] = predictAges
	age_isnull = age.loc[(train_data.Age.isnull())]
	# print(age_isnull)

	print('------------NoAges finished!-------------')

def Onehot():
	"""Onehot! Col: Sex, Embarked, (Drop) Cabin"""
	print('................Onehot!..................')
	train_data.loc[train_data['Sex'] == 'male', 'Sex'] = 0
	train_data.loc[train_data['Sex'] == 'female', 'Sex'] = 1
	train_data['Embarked'] = train_data['Embarked'].fillna('S')
	train_data.loc[train_data['Embarked'] == 'S', 'Embarked'] = 0
	train_data.loc[train_data['Embarked'] == 'C', 'Embarked'] = 1
	train_data.loc[train_data['Embarked'] == 'Q', 'Embarked'] = 2

	train_data.drop(['Cabin'], axis=1, inplace=True)
	train_data['Deceased'] = train_data['Survived'].apply(lambda s: 1-s)
	print('-------------Onehot finished!------------')

def train_func():
	dataset_x = train_data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]
	dataset_y = train_data[['Deceased', 'Survived']]
	x_train, x_val, y_train, y_val = train_test_split(dataset_x.values,
													dataset_y.values,
													test_size=0.1,
													random_state = 42)
	'''
	x = tf.placeholder(tf.float32, shape=[None, 6])
	y = tf.placeholder(tf.float32, shape=[None, 2])
	
	weight1 = tf.Variable(tf.random_normal([6, 6]))
	bias1 = tf.Variable(tf.zeros([6]))

	fc1 = tf.nn.relu(tf.add(tf.matmul(x, weight1), bias1))

	weight2 = tf.Variable(tf.random_normal([6, 2]))
	bias2 = tf.Variable(tf.zeros([2]))

	fc2 = tf.add(tf.matmul(fc1, weight2), bias2)

	model_train = tf.nn.softmax(fc2)
	'''
	x = tf.placeholder(tf.float32, shape=[None, 6])
	y = tf.placeholder(tf.float32, shape=[None, 2])

	weight1 = tf.Variable(tf.random_normal([6, 6]))
	bias1 = tf.Variable(tf.random_normal([6]))

	fc1 = tf.nn.relu(tf.add(tf.matmul(x, weight1), bias1))

	weigh2 = tf.Variable(tf.random_normal([6, 6]))
	bias2 = tf.Variable(tf.random_normal([6]))

	fc2 = tf.nn.relu(tf.add(tf.matmul(fc1, weigh2), bias2))

	weight3 = tf.Variable(tf.random_normal([6, 2]))
	bias3 = tf.Variable(tf.random_normal([2]))

	model_train = tf.nn.softmax(tf.add(tf.matmul(fc2, weight3), bias3))

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=model_train))
	correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(model_train, 1))
	acc_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	train_op = tf.train.AdamOptimizer(0.01).minimize(loss)

	saver = tf.train.Saver()

	ckpt_dir = path+'ckpt_dir'
	if not os.path.exists(ckpt_dir):
		os.makedirs(ckpt_dir)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		ckpt = tf.train.latest_checkpoint(ckpt_dir)
		if ckpt:
			print('Restoring from checkpoint: %s' % ckpt)
			saver.restore(sess, ckpt)
		
		for epoch in range(100):
			for i in range(len(x_train)):
				sess.run(train_op, feed_dict={x: [x_train[i]], y: [y_train[i]]})


			if epoch % 10 == 0:
				accuracy = sess.run(acc_op,feed_dict={x:x_val,y:y_val})
				print("Accuracy on validation set: %.9f" % accuracy)
				saver.save(sess, ckpt_dir + '/logistic.ckpt')
		print('training complete!')

		# 读测试数据  
		test_data = pd.read_csv(path+'test.csv')  

		#数据清洗, 数据预处理  
		test_data.loc[test_data['Sex']=='male','Sex'] = 0
		test_data.loc[test_data['Sex']=='female','Sex'] = 1 

		age = test_data[['Age','Sex','Parch','SibSp','Pclass']]
		age_notnull = age.loc[(test_data.Age.notnull())]
		age_isnull = age.loc[(test_data.Age.isnull())]
		X = age_notnull.values[:,1:]
		Y = age_notnull.values[:,0]
		rfr = RandomForestRegressor(n_estimators=1000,n_jobs=-1)
		rfr.fit(X,Y)
		predictAges = rfr.predict(age_isnull.values[:,1:])
		test_data.loc[(test_data.Age.isnull()),'Age'] = predictAges

		test_data['Embarked'] = test_data['Embarked'].fillna('S')
		test_data.loc[test_data['Embarked'] == 'S','Embarked'] = 0
		test_data.loc[test_data['Embarked'] == 'C','Embarked'] = 1
		test_data.loc[test_data['Embarked'] == 'Q','Embarked'] = 2

		test_data.drop(['Cabin'],axis=1,inplace=True)

		#特征选择
		X_test = test_data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]

		#评估模型
		predictions = np.argmax(sess.run(model_train, feed_dict={x: X_test}), 1)

		#保存结果
		submission = pd.DataFrame({
			"PassengerId": test_data["PassengerId"],
			"Survived": predictions
		})
		submission.to_csv("titanic-submission.csv", index=False)




if __name__ == '__main__':
	
	NoAges()
	Onehot()
	# train_data.to_csv(path+'deeled_train.csv', index=False)
	# print(train_data.info())
	train_func()
	


