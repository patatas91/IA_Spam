#!/usr/bin/env python

#
# Cristian Simon Moreno, 611487
#

######################################################
# Imports
######################################################

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
import numpy as np
import json
import glob
import sys
import os
import random
from sklearn import metrics
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics.classification import accuracy_score
#import time
#import matplotlib.patches as mpatches

######################################################
# Aux. functions
######################################################

# load_enron_folder: load training, validation and test sets from an enron path
def load_enron_folder(path):

   ### Load ham mails ###

   ham_folder = path + '/ham/*.txt'
   print("Loading files:", ham_folder)
   ham_list = glob.glob(ham_folder)
   num_ham_mails = len(ham_list)
   print num_ham_mails
   ham_mail = []
   for i in range(0,num_ham_mails):
      ham_i_path = ham_list[i]
      #print(ham_i_path)
      ham_i_file = open(ham_i_path, 'rb')  
      ham_i_str = ham_i_file.read()
      ham_i_text = ham_i_str.decode('utf-8',errors='ignore')     # Convert to Unicode
      ham_mail.append(ham_i_text)    # Append to the mail structure
      ham_i_file.close()
   random.shuffle(ham_mail)  # Random order

   # Separate into training, validation and test
   num_ham_training = int(round(0.8*num_ham_mails))
   ham_training_mail = ham_mail[0:num_ham_training]
   ham_training_labels = [0]*num_ham_training   
   #print(ham_training_labels)     

   num_ham_validation = int(round(0.1*num_ham_mails))
   ham_validation_mail = ham_mail[num_ham_training:num_ham_training+num_ham_validation]
   ham_validation_labels = [0]*num_ham_validation

   num_ham_test = num_ham_mails - num_ham_training - num_ham_validation
   ham_test_mail = ham_mail[num_ham_training+num_ham_validation:num_ham_mails]
   ham_test_labels = [0]*num_ham_test

   print("HAM mails       :", num_ham_mails)
   print("..for training  :", num_ham_training)
   print("..for validation:", num_ham_validation)
   print("..for testing   :", num_ham_test)

   ### Load spam mails ###

   spam_folder = path + '/spam/*.txt'
   print("Loading files:", spam_folder)
   spam_list = glob.glob(spam_folder)
   num_spam_mails = len(spam_list)
   spam_mail = []
   for i in range(0,num_spam_mails):
      spam_i_path = spam_list[i]
      #print(spam_i_path)
      spam_i_file = open(spam_i_path, 'rb')  
      spam_i_str = spam_i_file.read()
      spam_i_text = spam_i_str.decode('utf-8',errors='ignore')     # Convert to Unicode
      spam_mail.append(spam_i_text)    # Append to the mail structure
      spam_i_file.close()
   random.shuffle(spam_mail)  # Random order

   # Separate into training, validation and test
   num_spam_training = int(round(0.8*num_spam_mails))
   spam_training_mail = spam_mail[0:num_spam_training]
   spam_training_labels = [1]*num_spam_training

   num_spam_validation = int(round(0.1*num_spam_mails))
   spam_validation_mail = spam_mail[num_spam_training:num_spam_training+num_spam_validation]
   spam_validation_labels = [1]*num_spam_validation

   num_spam_test = num_spam_mails - num_spam_training - num_spam_validation
   spam_test_mail = spam_mail[num_spam_training+num_spam_validation:num_spam_mails]
   spam_test_labels = [1]*num_spam_test

   print("SPAM mails      :", num_spam_mails)
   print("..for training  :", num_spam_training)
   print("..for validation:", num_spam_validation)
   print("..for testing   :", num_spam_test)
   print "\n"

   ### spam + ham together ###
   training_mails = ham_training_mail + spam_training_mail
   training_labels = ham_training_labels + spam_training_labels
   validation_mails = ham_validation_mail + spam_validation_mail
   validation_labels = ham_validation_labels + spam_validation_labels
   test_mails = ham_test_mail + spam_test_mail
   test_labels = ham_test_labels + spam_test_labels

   data = {'training_mails': training_mails, 'training_labels': training_labels, 'validation_mails': validation_mails, 'validation_labels': validation_labels, 'test_mails': test_mails, 'test_labels': test_labels} 

   return data

# Split the data into k parts, hold one, combine and train the others parts. Validate it with the other part.
# Repeat K times holding a different part each time. Get the best results
def Kfold_cross_validation(classifier_type, k, mails, labels):
	best_size = 0 #LAPLACE
	best_score = 0.0
	best_accuracy = 0.0
	l = np.array(labels)
	print "(CROSS_VAL) > Kfold Cross Validation"
	print "(CROSS_VAL) > Calculating Laplace..."
	for i in range(0, 10):
		score = 0.0;
		accuracy = 0.0;
		k_fold = KFold(n=len(l), n_folds=k, shuffle=True, random_state=None)

		for train_i, test_i in k_fold:
			mail_train = mails[train_i]
			labels_train = l[train_i]
			mail_test = mails[test_i]
			labels_test = l[test_i]

			# Check type of classifier -> Multinomial o Bernouilli
			if (classifier_type == 'Multinomial'):
				''' BernouilliNB parameters: 
				-alpha: Laplace smoothing parameter
				-rest -> default
				'''
				classifier = BernoulliNB(alpha=i)
			else:
				''' MultinomialNB parameters: 
				-alpha: Laplace smoothing parameter
				-rest -> default
				'''
				classifier = MultinomialNB(alpha=i)

			# Train the model
			classifier.fit(mail_train, labels_train)
			# Predict the class of the mail
			prediction = classifier.predict(mail_test)

			# f1_score(valores_reales, valores_obtenidos_clasificador)
			# (2TP)/(2TP+FP+FN)
			score += f1_score(labels_test, prediction)
			# accuracy_score(valores_reales, valores_obtenidos_clasificador)
			# (TP+TN)/(P+N)
			accuracy += accuracy_score(labels_test, prediction)

		# Calculates score and accuracy means and check if better than the best
		score_mean = score/k
		accuracy_mean = accuracy/k
		if (accuracy_mean > best_accuracy):
			best_accuracy = accuracy_mean
		if (score_mean > best_score):
			best_score = score_mean
			best_size = i # LAPLACE

		# Results with K value
		#print "\n"
		#print "K value -> %d" % i
		#print "Score mean -> " + str(score_mean)
		#print "Accuracy mean -> " + str(accuracy_mean)
		#print "\n"

	# Best results of Kfold Cross Validation
	print "(CROSS_VAL) > -----------------------------------------"
	print "(CROSS_VAL) > Laplace -> " + str(best_size)
	print "(CROSS_VAL) > Best score -> " + str(best_score)
	print "(CROSS_VAL) > Best accuracy -> " + str(best_accuracy)
	print "(CROSS_VAL) > -----------------------------------------"

	return best_size

# Train the classifier with the model (bigrams/bag words) and 
# the classifier type (multinomial/Bernouilli)
def Train_classifier(train_data, labels, model, classifier_type):
	# Get best size -> Laplace (Kfold_cross_validation)
	laplace = Kfold_cross_validation(classifier_type, 9, train_data, labels)

	print "(TRAINER) > -----------------------------------------"
	print "(TRAINER) > " + classifier_type + " - " + model + " Naive Bayes" 
	print "(TRAINER) > -----------------------------------------"
	
	print "(TRAINER) > Laplace -> " + str(laplace)

	# Check type of classifier -> Multinomial o Bernouilli
	if (classifier_type == 'Multinomial'):
		''' BernouilliNB parameters: 
		-alpha: Laplace smoothing parameter
		-rest -> default
		'''
		classifier = BernoulliNB(alpha=laplace)
	else:
		''' MultinomialNB parameters: 
		-alpha: Laplace smoothing parameter
		-rest -> default
		'''
		classifier = MultinomialNB(alpha=laplace)

	print "(TRAINER) > Training classifier..."

	# Train the model
	classifier.fit(train_data, labels)
	# Predict the class of the mail
	prediction = classifier.predict(train_data)

	# f1_score(valores_reales, valores_obtenidos_clasificador)
	# (2TP)/(2TP+FP+FN)
	score = f1_score(labels, prediction)
	print "(TRAINER) > Score -> " + str(score)

	return (laplace, classifier)


# Show f1-score, confusion-matrix and precision-recall of all classifiers
# http://stackoverflow.com/questions/31421413/how-to-compute-precision-recall-accuracy-and-f1-score-for-the-multiclass-case
# http://stackoverflow.com/questions/26587759/plotting-precision-recall-curve-when-using-cross-validation-in-scikit-learn
def Evaluate_classifier(predictions, prediction_name, test_mails, test_labels):
	print "Starting..."
	best_f1score = 0
	best = 0
	# Dictionaries -> Precision-recall data
	precision = dict() 
	recall = dict()
	thresholds = dict()

	for i in range(0, len(predictions)):
		print "(EVALUATION) > -----------------------------------------"
		print "(EVALUATION) > prediction -> " + prediction_name[i]

		# F1-score		
		score = f1_score(test_labels, predictions[i])
		print "(EVALUATION) > F1-score = " + str(score)

		# Confusion matrix		
		confusion = confusion_matrix(test_labels, predictions[i])
		print "(EVALUATION) > Confusion matrix = "
		print confusion

		# Get figure params
		precision[i], recall[i], thresholds[i] = precision_recall_curve(test_labels, predictions[i])

		# Best classifier -> Best score
		if (score > best_f1score):
			best_f1score = score
			best = i

	# Precision-recall
	print "(EVALUATION) > PRINT precision Recall"
	# Delete previous figure
	plt.clf();
	# Plot figure
	# Multinomial Bag Words
	plt.plot(recall[0], precision[0], label='MBW')
	# Bernouilli Bag Words
	plt.plot(recall[1], precision[1], label='BBW')
	# Multinomial Bigrams
	plt.plot(recall[2], precision[2], label='MB')
	# Bernouilli Bigrams
	plt.plot(recall[3], precision[3], label='BB')

	# GRAFICA MAL
	plt.title('SPAM - HAM')
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.ylim([0.0, 1.05])
	plt.ylim([0.0, 1.0])
	plt.show()

	# Show best classifier
	print "(EVALUATION) > ========================================="
	print "(EVALUATION) > Best classifier : " + prediction_name[best]
	print "(EVALUATION) > -----------------------------------------"
	best_predictions = predictions[best]
	# Show f1-score
	score = f1_score(test_labels, best_predictions)
	print "(EVALUATION) > F1-score = " + str(score)
	# Show confusion matrix
	confusion = confusion_matrix(test_labels, best_predictions)
	print "(EVALUATION) > Confusion matrix = "
	print confusion
	# Build a text report showing the main classification metrics
	print "(EVALUATION) > Classification report: "
	print classification_report(test_labels, best_predictions, target_names=['0', '1'])
	print "(EVALUATION) > ========================================="

	# Classification ham-spam
	ham = [] # HAM mails
	spam = [] # SPAM mails
	false_ham = [] # Classificated as HAM but -> SPAM
	false_spam = [] # Classificated as SPAM but -> HAM

	for i in range(0, len(test_labels)):
		if (test_labels[i] == 0):
			if (best_predictions[i] == 0):
				ham.append(test_mails[i])
			else:
				false_spam.append(test_mails[i])
		else:
			if (best_predictions[i] == 0):
				false_ham.append(test_mails[i])
			else:
				spam.append(test_mails[i])
        
    # Show results
	#print "\n"
	print "(EVALUATION) > #########################################"
	print "(EVALUATION) > # HAM -> " + str(len(ham))
	print "(EVALUATION) > # SPAM -> " + str(len(spam))
	print "(EVALUATION) > # FALSE HAM -> " + str(len(false_ham))
	print "(EVALUATION) > # FALSE SPAM -> " + str(len(false_spam))
	print "(EVALUATION) > #########################################"

	# Example mails
	print "\n"
	print "Example mails..."
	print "(EVALUATION) > ========================================="
	print "(EVALUATION) > HAM: "
	print ham[0]
	print "\n"
	print "(EVALUATION) > ========================================="	
	print "(EVALUATION) > SPAM: "
	print spam[0]
	print "\n"
	print "(EVALUATION) > ========================================="	
	print "(EVALUATION) > FALSE HAM: "
	print false_ham[0]
	print "\n"
	print "(EVALUATION) > ========================================="	
	print "(EVALUATION) > FALSE SPAM: "
	print false_spam[0]	
	

######################################################
# Main
######################################################

# python -W ignore spam.py datasets/
print "Starting..."
# Get all files from arg[0] path
print ">> 1"
print "Checking files..."
args = sys.argv[1:]
folder = args[0]
print ">> " + folder
files = []
listing = os.listdir(folder)
for file in listing:
	print "-----------------------------------------"
	print "Found > " + file
	print "-----------------------------------------"
	data = load_enron_folder(folder+file)	
	files.append(data)
print "Done."
print "\n"

# Prepare data
print ">> 2"
print "Preparing data..."
# Training
training_mails = files[0]['training_mails']+files[1]['training_mails']+files[2]['training_mails']+files[3]['training_mails']+files[4]['training_mails']+files[5]['training_mails']
training_labels = files[0]['training_labels']+files[1]['training_labels']+files[2]['training_labels']+files[3]['training_labels']+files[4]['training_labels']+files[5]['training_labels']
# Validation
validation_mails = files[0]['validation_mails']+files[1]['validation_mails']+files[2]['validation_mails']+files[3]['validation_mails']+files[4]['validation_mails']+files[5]['validation_mails']
validation_labels = files[0]['validation_labels']+files[1]['validation_labels']+files[2]['validation_labels']+files[3]['validation_labels']+files[4]['validation_labels']+files[5]['validation_labels']
# Test
test_mails = files[0]['test_mails']+files[1]['test_mails']+files[2]['test_mails']+files[3]['test_mails']+files[4]['test_mails']+files[5]['test_mails']
test_labels = files[0]['test_labels']+files[1]['test_labels']+files[2]['test_labels']+files[3]['test_labels']+files[4]['test_labels']+files[5]['test_labels']
# Training + validation
trainval_mails = training_mails+validation_mails
trainval_labels = training_labels+validation_labels
print "Done."
print "\n"

# data predictions
data=[]
# name predictions
names=[]

# Evaluate models
print ">> 3"
print "Evaluate models..."

# Multinomial - Bag words
print "========================================="
print "Multinomial - Bag words"
print "-----------------------"
# Model -> Bag words - stop words
# Converting documents in descriptors of bag words
print "> Converting in descriptors..."
model = CountVectorizer(stop_words='english')
# Learn a vocabulary (fit)
print "> Learn vocabulary..."
mails_train = model.fit_transform(trainval_mails)
# Build the descriptors of the bag words
print "> Build the descriptors..."
mails_test = model.transform(test_mails)

# Train the classifier
print "> Training classifier..."
laplace, classifier = Train_classifier(mails_train, trainval_labels, "Bag words", 'Multinomial')

# Get the class wich an email belongs with his bag words with the trained classifier
print "> Making predictions..."
predictionMBW = classifier.predict(mails_test) 

# Save results
print "> Saving results..."
data.append(predictionMBW)
names.append('Multinomial - Bag words')
print "Done."
print "\n"

# Bernouilli - Bag words
print "========================================="
print "Bernouilli - Bag words"
print "----------------------"
# Model -> Bag words - stop words
# Converting documents in descriptors of bag words
print "> Converting in descriptors..."
model = CountVectorizer(stop_words='english')
# Learn a vocabulary (fit)
print "> Learn vocabulary..."
mails_train = model.fit_transform(trainval_mails)
# Build the descriptors of the bag words
print "> Build the descriptors..."
mails_test = model.transform(test_mails)

# Train the classifier
print "> Training classifier..."
laplace, classifier = Train_classifier(mails_train, trainval_labels, "Bag words", 'Bernouilli')

# Get the class wich an email belongs with his bag words with the trained classifier
print "> Making predictions..."
predictionBBW = classifier.predict(mails_test) 

# Save results
print "> Saving results..."
data.append(predictionBBW)
names.append('Bernouilli - Bag words')
print "Done."
print "\n"

# Multinomial - Bigrams
print "========================================="
print "Multinomial - Bigrams"
print "---------------------"
# Model -> Bigrams - stop words
# Converting documents in descriptors of bag words
print "> Converting in descriptors..."
model = CountVectorizer(ngram_range=(2,2), stop_words='english')
# Learn a vocabulary (fit)
print "> Learn vocabulary..."
mails_train = model.fit_transform(trainval_mails)
# Build the descriptors of the bag words
print "> Build the descriptors..."
mails_test = model.transform(test_mails)

# Train the classifier
print "> Training classifier..."
laplace, classifier = Train_classifier(mails_train, trainval_labels, "Bigrams", 'Multinomial')

# Get the class wich an email belongs with his bag words with the trained classifier
print "> Making predictions..."
predictionMB = classifier.predict(mails_test) 

# Save results
print "> Saving results..."
data.append(predictionMB)
names.append('Multinomial - Bigrams')
print "Done."
print "\n"

# Bernouilli - Bag words
print "========================================="
print "Bernouilli - Bigrams"
print "--------------------"
# Model -> Bigrams - stop words
# Converting documents in descriptors of bag words
print "> Converting in descriptors..."
model = CountVectorizer(ngram_range=(2,2), stop_words='english')
# Learn a vocabulary (fit)
print "> Learn vocabulary..."
mails_train = model.fit_transform(trainval_mails)
# Build the descriptors of the bag words
print "> Build the descriptors..."
mails_test = model.transform(test_mails)

# Train the classifier
print "> Training classifier..."
laplace, classifier = Train_classifier(mails_train, trainval_labels, "Bigrams", 'Bernouilli')

# Get the class wich an email belongs with his bag words with the trained classifier
print "> Making predictions..."
predictionBB = classifier.predict(mails_test) 

# Save results
print "> Saving results..."
data.append(predictionBB)
names.append('Bernouilli - Bigrams')
print "Done."
print "\n"

# Results of all classifiers
print ">> 4"
print "========================================="
print "Classifiers evaluation"
print "========================================="
Evaluate_classifier(data, names, test_mails, test_labels)