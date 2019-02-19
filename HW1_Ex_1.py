# -*- coding: utf-8 -*-
"""
Data Mining Homework 1 - dimensionality reduction

@author Tobias Braun, tgb2117
"""
################################imports########################################
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.linalg as splin
np.random.seed(100)
############################prerequisites######################################
x_array = np.array(np.round_(np.random.rand(16)*100) - 50)
y_array = np.array(np.round_(np.random.rand(16)*100) - 50)

assert (np.abs(np.dot(x_array, y_array)) > 1)

################dimensionality reduction methods - definition##################
def IID (x: np.array, m: int, seed=100):
    np.random.seed(seed)
    G = np.random.normal(size=(m, np.shape(x)[0]))
    return np.matmul(G/np.sqrt(m), x)


def CIRC (x: np.array, m: int, seed=100):
    np.random.seed(seed)
    G = np.random.normal(size=(np.shape(x)[0]))
    CIRC = G
    for i in range(1, m):
        CIRC = np.vstack((CIRC, np.roll(G, 1)))
        G = np.roll(G, 1)
    return np.matmul(CIRC/np.sqrt(m), x)


def GORT (x: np.array, m: int, seed=100):
    np.random.seed(seed)
    G = np.random.normal(size=(np.shape(x)[0], m))
    GORT = np.transpose(np.linalg.qr(G)[0]*np.sqrt(np.shape(x)[0]))
    return np.matmul(GORT/np.sqrt(m), x)


def HD (x: np.array, m: int, seed=100):
    np.random.seed(seed)
    H = splin.hadamard(np.shape(x)[0])
    vec = np.random.choice([-1,1], np.shape(x)[0])
    D = np.diag(vec)
    vec_2 = np.random.choice([-1,1], np.shape(x)[0])
    D_2 = np.diag(vec_2)
    vec_3 = np.random.choice([-1,1], np.shape(x)[0])
    D_3 = np.diag(vec_3)
    
    HD_1 = np.matmul(H, D)
    HD_2 = np.matmul(H, D_2)
    HD_3 = np.matmul(H, D_3)
    HD_final = np.matmul(HD_1, HD_2)
    HD_final = np.matmul(HD_final, HD_3)/np.shape(x)[0]
    return np.matmul(HD_final/np.sqrt(m), x)

def KAC(x: np.array, m: int, seed=100):
    np.random.seed(seed)
    dim = np.shape(x)[0]
    no_of_givens = np.ceil(dim*np.log(dim))
    
    def givens (dim: int, seed=100):
        np.random.seed(seed)
        vec = np.repeat(1.0, dim)
        D = np.diag(vec)
        indices = np.random.choice(np.arange(0, dim), size=2, replace=False)
        i = np.amin(indices)
        j = np.amax(indices)
        theta = np.random.rand()*2*np.pi
        D[i][i] = np.cos(theta)
        D[i][j] = np.sin(theta)
        D[j][i] = -np.sin(theta)
        D[j][j] = np.cos(theta)
        return D
    
    multi_givens = [givens(dim, i) for i in np.random.choice(
            np.arange(0,100000), size=int(no_of_givens), replace=False)]
    pre_KAC = np.identity(dim)
    for i in range(0, len(multi_givens)):
        pre_KAC = np.matmul(pre_KAC, multi_givens[i])
        
    indices = np.random.choice(np.arange(0, dim), size=m, replace=False)
    KAC = np.sqrt(dim)*pre_KAC[indices]
    
    return np.matmul(KAC/np.sqrt(m), x)
    

###############dimensionality reduction methods - MSE testing##################

true_kernel_value = np.dot(x_array, y_array)
m_ = [1,2,4,6,8,10,12,14]

def MSE_Estimate_IID(x: np.array, y: np.array, m: int):
    iid = 0
    for i in range(0,1000):
        x_red = IID(x, m, i)
        y_red = IID(y, m, i)
        kernel_after_d_reduction = np.dot(x_red, y_red)
        sq_diff = np.power(kernel_after_d_reduction - true_kernel_value, 2)
        iid += sq_diff
        
    MSE_estimate = iid/1000
    return MSE_estimate

IID_MSE_Estimate = [MSE_Estimate_IID(x_array, y_array, m) for m in m_]

def MSE_Estimate_CIRC(x: np.array, y: np.array, m: int):
    circ = 0
    for i in range(0,1000):
        x_red = CIRC(x, m, i)
        y_red = CIRC(y, m, i)
        kernel_after_d_reduction = np.dot(x_red, y_red)
        sq_diff = np.power(kernel_after_d_reduction - true_kernel_value, 2)
        circ += sq_diff
        
    MSE_estimate = circ/1000
    return MSE_estimate

CIRC_MSE_Estimate = [MSE_Estimate_CIRC(x_array, y_array, m) for m in m_]

def MSE_Estimate_GORT(x: np.array, y: np.array, m: int):
    gort = 0
    for i in range(0,1000):
        x_red = GORT(x, m, i)
        y_red = GORT(y, m, i)
        kernel_after_d_reduction = np.dot(x_red, y_red)
        sq_diff = np.power(kernel_after_d_reduction - true_kernel_value, 2)
        gort += sq_diff
        
    MSE_estimate = gort/1000
    return MSE_estimate

GORT_MSE_Estimate = [MSE_Estimate_GORT(x_array, y_array, m) for m in m_]

def MSE_Estimate_HD(x: np.array, y: np.array, m: int):
    hd = 0
    for i in range(0,1000):
        x_red = HD(x, m, i)
        y_red = HD(y, m, i)
        kernel_after_d_reduction = np.dot(x_red, y_red)
        sq_diff = np.power(kernel_after_d_reduction - true_kernel_value, 2)
        hd += sq_diff
        
    MSE_estimate = hd/1000
    return MSE_estimate

HD_MSE_Estimate = [MSE_Estimate_HD(x_array, y_array, m) for m in m_]

def MSE_Estimate_KAC(x: np.array, y: np.array, m: int):
    kac = 0
    for i in range(0,1000):
        x_red = KAC(x, m, i)
        y_red = KAC(y, m, i)
        kernel_after_d_reduction = np.dot(x_red, y_red)
        sq_diff = np.power(kernel_after_d_reduction - true_kernel_value, 2)
        kac += sq_diff
        
    MSE_estimate = kac/1000
    return MSE_estimate

KAC_MSE_Estimate = [MSE_Estimate_KAC(x_array, y_array, m) for m in m_]


IID_ = plt.plot(m_, IID_MSE_Estimate, label='IID')
CIRC_ = plt.plot(m_, CIRC_MSE_Estimate, label='CIRC')
GORT_ = plt.plot(m_, GORT_MSE_Estimate, label='GORT')
HD_ = plt.plot(m_, HD_MSE_Estimate, label='HD')
KAC_ = plt.plot(m_, KAC_MSE_Estimate, label='KAC')
plt.legend()
plt.title("MSE Estimates for different padding mechanisms")
plt.xlabel("m")
plt.ylabel("MSE")
#plt.savefig('MSE_Estimates.png')
plt.show()

###############################conclusion######################################

'''When only considering the MSE of each dimensionalty reduction method it appears
that IID is worst, then CIRC, then GORT, then KAC and the winner is the HD method 
using 3 HD blocks. One should always keep in mind that unbiasedness but even more 
so complexity are important factors when deciding which method to choose. 
Obviously, the accuracy of all estimators increases drastically with m and the 
MSE approaches 0 when m approaches d, which in our case is 16. But as the whole 
purpose of dimensionality reduction is to reduce m, this effect is not so much of
relevance when comparing different estimators.'''
