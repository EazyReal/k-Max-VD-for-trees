#!/usr/bin/env python
# coding: utf-8
# author: Yan-Tong Lin

import numpy as np
from collections import defaultdict
import networkx as nx
import time
import subprocess
from subprocess import Popen, PIPE, STDOUT, run
import  matplotlib.pyplot as plt

exe_file_name = "exp_cpp"
exe_cmd = "./" + exe_file_name


def str_tc(n, k):
    T = nx.generators.trees.random_tree(n, int(time.time()))
    # print(T.edges)
    tc = ""
    tc += str(n) + " " + str(k)
    for a, b in T.edges:
        tc += " " + str(a) + " " + str(b)
    return tc


def run1(n, k):
    p = run([exe_cmd], stdout=PIPE, input=str_tc(n,k), encoding='ascii')
    ans, t= map(int,p.stdout.split())
    return ans, t

def exp(range_n=100, range_split=10, exp_time=10):
    dat = np.zeros((range_n//range_split+1, range_n//range_split+1, exp_time))
    for exp_n in range(1,range_n+1, range_split):
        for exp_k in range(1, exp_n+1, range_split):
            for ti in range(exp_time):
                ans, tt = run1(exp_n, exp_k)
                dat[exp_n//range_split][exp_k//range_split][ti] = tt
    return dat
    
dat2 = exp(1000,100,5)


def show_stat(dat, range_n, range_split):
    dats = dat.sum(axis=2)
    #print(dats)
    
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    for i in range(range_n//range_split):
        ax1.set_title("n-k graph")
        ax1.plot(dats[i][:i], label=("n = " + str(i*range_split+1)))
        ax1.legend()
        
    for i in range(range_n//range_split):
        #print(i, dats.T[i][i:-1])
        ax2.set_title("k-n graph")
        ax2.plot((dats.T[i][i:-1]), label=("k = " + str(i*range_split+1)))
        ax2.legend()
    

show_stat(dat2, 1000, 100)