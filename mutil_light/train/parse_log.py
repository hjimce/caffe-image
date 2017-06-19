#coding=utf-8
import numpy as np
import re
import click
import glob, os
from matplotlib import pylab as plt
import operator
import ntpath
import pickle
def main(files):
    for f in files:
        deleteline(f)
    plt.style.use('ggplot')
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('loss')
    ax2.set_ylabel('accuracy %')
    if not files:
        print 'no args found'
        print '\n\rloading all files with .log extension from current directory'
        os.chdir(".")
        files = glob.glob("*.log")

    for i, log_file in enumerate(files):
        loss_iterations, losses, accuracy_iterations, accuracies, accuracies_iteration_checkpoints_ind, fileName = parse_log(log_file)
        disp_results(fig, ax1, ax2, loss_iterations, losses, accuracy_iterations, accuracies, accuracies_iteration_checkpoints_ind, fileName, color_ind=i+2)
    plt.show()

def deleteline(log_file,substr='blocking_queue.cpp'):
    print log_file
    lines=[]
    with open(log_file, 'r') as f:
        for l in f.readlines():
            if substr in l:
                continue
            lines.append(l)
    print lines
    with open(log_file, 'w') as f:
        for l in lines:
            f.writelines(l)




def parse_log(log_file):
    with open(log_file, 'r') as log_file2:
        log = log_file2.read()

    loss_pattern = r"Iteration (?P<iter_num>\d+), loss = (?P<loss_val>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    losses = []
    loss_iterations = []

    fileName= os.path.basename(log_file)
    for r in re.findall(loss_pattern, log):
        loss_iterations.append(int(r[0]))
        losses.append(float(r[1]))

    loss_iterations = np.array(loss_iterations)
    losses = np.array(losses)

    accuracy_pattern = r"Iteration (?P<iter_num>\d+), Testing net \(#0\)\n.* accuracy = (?P<accuracy>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    accuracies = []
    accuracy_iterations = []
    accuracies_iteration_checkpoints_ind = []

    for r in re.findall(accuracy_pattern, log):
        iteration = int(r[0])
        accuracy = float(r[1]) * 100

        if iteration % 10000 == 0 and iteration > 0:
            accuracies_iteration_checkpoints_ind.append(len(accuracy_iterations))

        accuracy_iterations.append(iteration)
        accuracies.append(accuracy)

    accuracy_iterations = np.array(accuracy_iterations)
    accuracies = np.array(accuracies)

    return loss_iterations, losses, accuracy_iterations, accuracies, accuracies_iteration_checkpoints_ind, fileName


def disp_results(fig, ax1, ax2, loss_iterations, losses, accuracy_iterations, accuracies, accuracies_iteration_checkpoints_ind, fileName, color_ind=0):
    color=['r','g','b','y']
    acrIterations =[]
    top_acrs={}
    if accuracies.size:
        if 	accuracies.size>4:
		    top_n = 4
        else:
            top_n = accuracies.size -1
        temp = np.argpartition(-accuracies, top_n)
        result_indexces = temp[:top_n]
        temp = np.partition(-accuracies, top_n)
        result = -temp[:top_n]
        for acr in result_indexces:
            acrIterations.append(accuracy_iterations[acr])
            top_acrs[str(accuracy_iterations[acr])]=str(accuracies[acr])

        sorted_top4 = sorted(top_acrs.items(), key=operator.itemgetter(1))
        maxAcc = np.amax(accuracies, axis=0)
        iterIndx = np.argmax(accuracies)
        maxAccIter = accuracy_iterations[iterIndx]
        maxIter =   accuracy_iterations[-1]
        consoleInfo = format('\n[%s]:maximum accuracy [from 0 to %s ] = [Iteration %s]: %s ' %(fileName,maxIter,maxAccIter ,maxAcc))
        plotTitle = format('max accuracy(%s) [Iteration %s]: %s ' % (fileName,maxAccIter, maxAcc))
        print (consoleInfo)

        print ('Top 4 accuracies:'+str(sorted_top4))
        plt.title(plotTitle)
    ax1.plot(loss_iterations, losses, color=color[color_ind])
    ax2.plot(accuracy_iterations, accuracies,color=color[color_ind], label=str(fileName))
    ax2.plot(accuracy_iterations[accuracies_iteration_checkpoints_ind], accuracies[accuracies_iteration_checkpoints_ind], 'o', color=color[color_ind])
    plt.legend(loc='lower right')


main(['log.txt','log13.txt'])

#main(['log_old.txt'])

