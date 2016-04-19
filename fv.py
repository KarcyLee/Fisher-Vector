#!/usr/bin/python27
#encoding=utf-8
import sys, argparse
from numpy import *
from sklearn import mixture
import math
import time
from scipy.stats import multivariate_normal

def read_features(txt,cols=100):
	features=[]
	f = open(txt)
	lines = f.readlines();
	for line in lines:
		line_feature=[]
		vec = line.split(' ')
		if size(vec) != cols:
			print "cols of input is wrong"
		else:
		#print size(vec)
			for v in vec:
				try:
					line_feature.append(float32(v))
				except ValueError:
					print 'invalid input %s' %(v)
			features.append(line_feature)
	f.close()
	features = array(features)
	return features

def dictionary(descriptors, N):
	gmm = mixture.GMM(n_components=N,covariance_type='full')
	gmm.fit(descriptors)
	#save("means.gmm", gmm.means_)
	#save("covs.gmm", gmm.covars_)
	#save("weights.gmm", gmm.weights_)
	return float32(gmm.means_),float32(gmm.covars_),float32(gmm.weights_)

def likelihood_moment(x, gaussians, weights, k, moment):	
	x_moment = power(float32(x), moment) if moment > 0 else float32([1])
	probabilities = map(lambda i: weights[i] * gaussians[i], range(0, len(weights)))
	ytk = probabilities[k] / sum(probabilities)
	return x_moment * ytk

def likelihood_statistics(samples, means, covs, weights):
	s0, s1,s2 = {}, {}, {}
	samples = zip(range(0, len(samples)), samples)
	gaussians = {}
	g = [multivariate_normal(mean=means[k], cov=covs[k]) for k in range(0, len(weights)) ]
	for i,x in samples:
		gaussians[i] = {k : g[k].pdf(x) for k in range(0, len(weights) ) }

	for k in range(0, len(weights)):
		s0[k] = reduce(lambda a, (i,x): a + likelihood_moment(x, gaussians[i], weights, k, 0), samples, 0)
		s1[k] = reduce(lambda a, (i,x): a + likelihood_moment(x, gaussians[i], weights, k, 1), samples, 0)
		s2[k] = reduce(lambda a, (i,x): a + likelihood_moment(x, gaussians[i], weights, k, 2), samples, 0)
	return s0, s1, s2

def fisher_vector_weights(s0, s1, s2, means, covs, w, T):
	return float32([((s0[k] - T * w[k]) / sqrt(w[k]) ) for k in range(0, len(w))])

def fisher_vector_means(s0, s1, s2, means, sigma, w, T):
	return float32([(s1[k] - means[k] * s0[k]) / (sqrt(w[k] * sigma[k])) for k in range(0, len(w))])

def fisher_vector_sigma(s0, s1, s2, means, sigma, w, T):
	return float32([(s2[k] - 2 * means[k]*s1[k]  + (means[k]*means[k] - sigma[k]) * s0[k]) / (sqrt(2*w[k])*sigma[k])  for k in range(0, len(w))])

def normalize(fisher_vector):
	v = sqrt(abs(fisher_vector)) * sign(fisher_vector)
	return v / sqrt(dot(v, v))


"""
************************Interface*****************************************
"""


def generate_gmm(gmm_folder,descriptors, N):
	'''
	Interface
	gmm_folder
	descriptors, (Train data) ,numpy.array, matrix, each row is one sample
	N, int ,the number of cluster center
	'''
	means,covs,weights = dictionary(descriptors,N)
	save(gmm_folder+ '/'+ "means.gmm", means)
	save(gmm_folder+ '/'+ "covs.gmm", covs)
	save(gmm_folder+ '/'+ "weights.gmm", weights)
	return means, covs, weights	

def load_gmm(folder = "."):
	'''
	Interface
	'''
	files = ["means.gmm.npy", "covs.gmm.npy", "weights.gmm.npy"]
	return map(lambda file: load(open(file,'rb')), map(lambda s : folder + "/" + s , files))

def fisher_vector(samples, means, covs, w):
	'''
	Interface: 
	samples: (to be encoded ),numpy.array , matrix, each row is a sample
	means: gmm.means_
	covs: gmm.covars_
	w: gmm.weights_
	'''
	s0, s1, s2 =  likelihood_statistics(samples, means, covs, w)
	T = samples.shape[0]
	covs = float32([diagonal(covs[k]) for k in range(0, covs.shape[0])])
	a = concatenate(fisher_vector_weights(s0, s1, s2, means, covs, w, T))
	a = normalize(a)
	b = concatenate(fisher_vector_means(s0, s1, s2, means, covs, w, T))
	b = normalize(b)
	c = concatenate(fisher_vector_sigma(s0, s1, s2, means, covs, w, T))
	c = normalize(c)
	fv = concatenate([a,b,c])
	#fv = concatenate([concatenate(a), concatenate(b), concatenate(c)])
	#fv = normalize(fv)
	return fv



