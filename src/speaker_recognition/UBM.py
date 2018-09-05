#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 23:12:30 2018
@author: siva
"""
import numpy as np
import csv
from utils import unit_gaussian
from k_means import k_means
import pickle
from scipy.stats import multivariate_normal

def special_op(array_1,array_2,D,N):
    for n in range(N):
        array_1[:,n] *= array_2[n]
    return array_1

def train_ubm(args):
    N = args.N
    D = args.D
    K = args.K
    iterations = 0
    with open(args.mfcc_file,"rb") as file:
        data=pickle.load(file)
    if N>data.shape[0]:
        N=data.shape[0]
    data=np.array(data[:N])

    for_cov = []
    for d in range(D):
        for_cov.append(data[:,d])
    
    p_nk = np.zeros((N,K))
    mu_k = np.zeros((K,D))
    cov_k = np.zeros((K,D,D))
    pi_k = np.zeros((K,1))
    #p_k = np.zeros((K,1))


    
    #mu_k = np.random.randn(K,D)
    #cov_k = np.array([np.cov(for_cov)]*K)
    #pi_k = np.reshape(np.array([0.5]*K),(K,1))
    mu_k,cov_k,pi_k = k_means(data,N,K,D,100)
    cov_k += 0.001*np.eye(D)
    
    def plot(z_n_k):
        plt.figure()
        color_array = ["r","g","b","y","black"]
        for i,each in enumerate(data):
            plt.scatter(each[0],each[1],c = color_array[np.argmax(z_n_k[i,:])],edgecolor = color_array[np.argmax(z_n_k[i,:])])
        plt.show()
    
    def m_step():
        p_k = np.sum(p_nk,axis = 0)
        #print("--->",p_k)
        for k in range(K):
            s = np.zeros((1,D))
            for n in range(N):
                s += p_nk[n][k]*data[n,:]
            mu_k[k] = (1/p_k[k])*s
        #print("--mu",mu_k)
        for k in range(K):
            #siva's tmp cov
            #cov_k[k] = np.dot(np.dot((data - mu_k[k,:]).T, np.diag(p_nk[:,k])), (data-mu_k[k])) / p_k[k]
            #full cov
            #cov_k[k] = np.dot(special_op((data - mu_k[k,:]).T,(p_nk[:,k]),D,N), (data-mu_k[k])) / p_k[k]
            #diag cov
            diag=np.zeros(D)
            for d in range(D):
                for n in range(N):
                    diag[d]+=p_nk[n,k]*(data[n,d]-mu_k[k,d])**2
                diag[d]/=p_k[k]
            cov_k[k]=np.diag(diag)
            cov_k[k] = cov_k[k] + 0.001*np.eye(cov_k[k].shape[0])
        #print("--cov",cov_k)
        for k in range(K):
            pi_k[k] = p_k[k]/N
        #print("--pi",pi_k)
        
    def e_step():
        num = np.zeros((K,1))
        #print(cov_k)
        for n in range(N):
            for k in range(K):
                #num[k] = pi_k[k] * (multivariate_normal.pdf(data[n],mu_k[k],cov_k[k]))
                num[k] = pi_k[k] * (unit_gaussian(data[n],mu_k[k],cov_k[k]))
            p_nk[n] = np.reshape(num/np.sum(num),(K,))
        #print("-->z",p_nk[n])
        
    def calculate_likelihood():
        log_likelihood = 0
        for n in range(N):
            temp = 0
            for k in range(K):
                #temp += pi_k[k] * (multivariate_normal.pdf(data[n],mu_k[k],cov_k[k]))
                temp += pi_k[k,0] * (unit_gaussian(data[n],mu_k[k],cov_k[k]))
            log_likelihood += np.log(temp)
        return log_likelihood

    old_likelihood = 0
    new_likelihood = 9999
    likelihood = []

    while(abs(old_likelihood - new_likelihood)>args.likelihood_threshold and iterations < args.max_iterations):
        iterations += 1
        old_likelihood = new_likelihood
        e_step()
        m_step()
        new_likelihood = calculate_likelihood()
        likelihood.append(new_likelihood)
        print("iterations:{0}|likelihood:{1}".format(str(iterations),str(new_likelihood)))
        #plot(p_nk)
        
    print(likelihood)
    
    final_dict = {"mean":mu_k,"cov":cov_k,"pi":pi_k,"likelihood":likelihood}
    with open(args.ubm_file_name, "wb") as f:
        pickle.dump(final_dict,f)
    
    #load final _dict:
#    with open(arguments.ubm_file_name, "rb") as f:
#        final_dict=pickle.load(f)
#    mu_k=final_dict['mean']
#    cov_k=final_dict['cov']
#    pi_k=final_dict['pi']
    
    np.set_printoptions(threshold=np.nan,suppress=True)
    with open('Mean.txt', 'w') as f:
        f.writelines(np.array2string(x,max_line_width=99999,separator=',')+'\n' for x in mu_k)
    with open('Pi.txt', 'w') as f:
        f.writelines(np.array2string(x,max_line_width=99999,separator=',')+'\n' for x in pi_k)
#    with open('ubm_cov.txt', 'w') as f:
#        f.write(np.array2string(pi_k,max_line_width=99999))
    a=dict()
    for i in range(256):
        a[str(i)]=[list(x) for x in cov_k[i]]
    with open('Cov.json', 'w') as f:
       json.dump(a,f,sort_keys=True,indent=2)