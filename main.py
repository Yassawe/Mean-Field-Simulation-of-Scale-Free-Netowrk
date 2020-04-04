import random
from numpy import seterr
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
seterr(all='ignore')

class Simulation():
    def __init__(self, N, gamma, lam, lam_end, step):
        self.N = N
        self.lam = lam
        self.lam_end = lam_end
        self.step = step
        zeta1, zeta2 = 0, 0
        for i in range(1, self.N):
            a = i ** (-gamma)
            zeta1 += a
            zeta2 += i * a
        self.k_mean = (1 + zeta2) / (1 + zeta1)
        self.prob_distribution = [0, 0]
        for i in range(2, N):
            self.prob_distribution.append(i ** (-gamma) / (zeta1 - 1))

    def model1a(self, theta):
        summation = 0
        for i in range(self.N):
            summation += i * self.prob_distribution[i] * (i * self.lam) / (self.k_mean + i * self.lam * theta)
        return summation - self.k_mean

    def model1b(self, theta):
        summation = 0
        for i in range(self.N):
            summation += i*self.prob_distribution[i]*self.lam*(1-(1-theta)**i)/((self.k_mean+self.lam*(1-(1-theta)**i))*self.k_mean)
        return summation - theta

    def model1c(self, theta):
        summation=0
        for i in range(self.N):
            summation+=i*self.prob_distribution[i]*self.lam*(i*theta+1-(1-theta)**i)/(2*self.k_mean + self.lam*(i*theta+1-(1-theta)**i)*self.k_mean)
        return summation - theta

    def model2a(self, p):
        summation=0
        for i in range(self.N):
            summation += i*self.prob_distribution[i]*self.lam/(self.k_mean+i*self.lam*p)
        return summation - 1

    def model2c(self, var):
        theta, p = var
        summation1=0
        summation2=0
        for i in range(self.N):
            summation1+=i*self.prob_distribution[i]*self.lam*(theta+i*p/self.k_mean)/((2+self.lam*(theta+i*p/self.k_mean))*self.k_mean)
            summation2+=self.prob_distribution[i]*self.lam*(theta+i*p/self.k_mean)/(2+self.lam*(theta+i*p/self.k_mean))
        return summation1-theta, summation2-p


    def generateEquilibriumDensity(self, model):
        equilibriumDensity = 0
        if model == 1:
            theta=fsolve(self.model1a, random.random())
            for i in range(self.N):
                p_k = i*self.lam*theta[0]/(self.k_mean+i*self.lam*theta[0])
                equilibriumDensity += p_k*self.prob_distribution[i]
        elif model == 2:
            theta=fsolve(self.model1b, random.random())
            for i in range(self.N):
                p_k = self.lam*(1-(1-theta[0])**i)/(self.k_mean + self.lam*(1-(1-theta[0])**i))
                equilibriumDensity += p_k*self.prob_distribution[i]
        elif model == 3:
            theta = fsolve(self.model1c, random.random())
            for i in range(self.N):
                p_k = self.lam*(i*theta[0]+1-(1-theta[0])**i)/(2*self.k_mean + self.lam*(i*theta[0]+1-(1-theta[0])**i))
                equilibriumDensity += p_k*self.prob_distribution[i]
        elif model == 4:
            equilibriumDensity = fsolve(self.model2a, random.random())
        elif model == 5:
            equilibriumDensity = 1-1/self.lam
        elif model == 6:
            # noinspection PyTypeChecker,PyTupleAssignmentBalance
            theta, equilibriumDensity = fsolve(self.model2c, (random.random(), random.random()))


        if equilibriumDensity>1:
            equilibriumDensity = 1
        elif equilibriumDensity<0:
            equilibriumDensity = 0

        return equilibriumDensity

    def generatePlot(self, model):
        l=[]
        p=[]
        while self.lam<=self.lam_end:
            l.append(self.lam)
            p.append(self.generateEquilibriumDensity(model))
            self.lam+=self.step
        self.lam=l[0]
        return [l, p]


def average(num_iterations, sim, model):
    arr=[]
    for i in range(num_iterations):
        arr.append(sim.generatePlot(model))
    l=arr[0][0]
    p=[]
    for i in range(len(arr)):
        for j in range(len(arr[0][1])):
            if len(p)!=len(arr[0][1]):
                p.append(arr[i][1][j])
            else:
                p[j]+=arr[i][1][j]
    for i in range(len(p)):
        p[i]=p[i]/len(arr)
    return [l,p]

def driver(num_iterations=1):
    sim1 = Simulation(1000, 2.25, 0.01, 4, 0.1)
    sim2 = Simulation(1000, 2.75, 0.01, 4, 0.1)

    for i in range(1,7):  # from 1 to 6
        plt.subplot(2, 3, i)
        plt.ylim(0, 1)
        [l1, p1] = average(num_iterations,sim1,i)
        [l2, p2] = average(num_iterations,sim2,i)
        plt.plot(l1, p1)
        plt.plot(l2, p2)

    plt.show()

driver(10)




