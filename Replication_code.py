#Coupled Learning (Puranam and Swamy, 2016)

#Importing modules
#from IPython import get_ipython
#get_ipython().magic('reset -sf')
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import csv
import datetime


STARTINGTIME = datetime.datetime.now().replace(microsecond=0)

#####################################################################################################
# SET SIMULATION PARAMETERS HERE
T=100#number of periods to simulate the model
NP=5000#number of pairs of agents
Pi_max = 1 #Payoff for the optimal action
Pi_min = -1 #Payoff for suboptimal actions

# Task environments
M=10#the number of possible actions

# Initial representation
IR_1 = 1 # 0 = Uniform initial representation; 1 = Good initial representation; 2 = bad initial representation;
IR_2 = 2 # 0 = Uniform initial representation; 1 = Good initial representation; 2 = bad initial representation

#agent learning parameters
p = 0.8 #Strength of beliefs
tau = 0.1 #Exploration rate
phi = 0.25 #Learning rate
att1=np.zeros(M) # Belief of agent 1
att2=np.zeros(M) # Belief of agent 2
######################################################################################################

#Defining functions
def environment(m):#Construct task environments
    r = Pi_min * np.ones((m,m))
    r[0, 0] = Pi_max
    return r

def softmax(attraction,dim): #softmax action selection
    prob=np.zeros(dim)
    denom=0
    i=0
    while i<dim:
        denom=denom + math.exp((attraction[i])/tau)
        i=i+1
    roulette=random.random()
    i=0
    probability=0
    while i<dim:
        prob[i]=math.exp(attraction[i]/tau)/denom
        probability = probability + prob[i]
        if probability>roulette:
            choice = i
            return choice
            break #stops computing probability of action selection as soon as cumulative probability exceeds roulette
        i=i+1

def initialrepresentation(ir):
    r = (Pi_min * p + Pi_max * (1-p)) * np.ones(M)
    if ir == 1:
        r[0] = Pi_min * (1 - p) + Pi_max * p
    elif ir == 2:
        r[1] = Pi_min * (1 - p) + Pi_max * p
    return r

#SIMULTAION IS RUN HERE
E = environment(M)
result_org=np.zeros((T,3))

for a in range(NP):
    att1 = initialrepresentation(IR_1)
    att2 = initialrepresentation(IR_2)
    for t in range(T):
        result_org[t, 0] = t
        action1 = softmax(att1, M)
        action2 = softmax(att2, M)

        if action1 == 0 and action2 == 0:
            att1[action1] = (1-phi)*att1[action1] + phi*Pi_max
            att2[action2]  = (1-phi)*att2[action2] + phi*Pi_max
            result_org[t, 1] += Pi_max/NP
            result_org[t, 2] += 1/NP
        else:
            att1[action1] = (1-phi)*att1[action1] + phi*Pi_min
            att2[action2]  = (1-phi)*att2[action2] + phi*Pi_min
            result_org[t, 1] += Pi_min / NP

#WRITING RESULTS TO CSV FILE   
filename = ("Coupled Learning"+"_Phi="+str(phi) + "_Tau=" + str(tau)+"_IR1="+str(IR_1)+"_IR2="+str(IR_2)+'.csv')
with open (filename,'w',newline='')as f:
    thewriter=csv.writer(f)
    thewriter.writerow(['Period', 'Performance', 'Probability of the optimal action'])
    for values in result_org:
        thewriter.writerow(values)
    f.close()  


##PRINTING END RUN RESULTS
print ("Final Performance"+str(result_org[T-1,1]))

ENDINGTIME = datetime.datetime.now().replace(microsecond=0)
TIMEDIFFERENCE = ENDINGTIME - STARTINGTIME
#print 'Computation time:', TIMEDIFFERENCE    
    
