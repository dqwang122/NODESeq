# import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plotlosscurve(loss, name, savename):
    plt.figure()
    print(len(loss))
    if isinstance(name, list):
        for x,n in zip(loss, name):
            plt.plot(range(len(x)), x, label=n)
        plt.title("Convergence Graph of {} Cost Function".format('/'.join(name)))
    else:
        plt.plot(range(len(loss)), loss, '-')
        plt.title("Convergence Graph of {} Cost Function".format(name))
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.legend()
    plt.savefig(savename, dpi=200)

def plottimecurve(time, name, savename):
    plt.figure()
    plt.plot(range(len(time)), np.cumsum(time), 'r')
    plt.title("Training time for {} ".format(name))
    plt.xlabel("Number of Iterations")
    plt.ylabel("Wall Clock Time")
    plt.savefig(savename, dpi=200)

def plotnfecurve(nfe, time, name, savename):
    plt.figure()
    plt.plot(time, nfe, 'r')
    plt.title("Number of evaluation {} training time".format(name))
    plt.xlabel("Wall Clock Time")
    plt.ylabel("NFE")
    plt.savefig(savename, dpi=200)