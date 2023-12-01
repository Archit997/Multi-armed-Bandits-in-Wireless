import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import time
class MultiArmedBandit:
    def __init__(self, size, explore_rounds, exploit_rounds):
        self.size = size
        self.success_probabilities = np.random.uniform(0,1,size)
        self.means_array = np.zeros(self.size, dtype=float)
        self.explore_rounds = explore_rounds
        self.exploit_rounds = exploit_rounds
        self.cost=0
        self.sorted_prob_exploration = self.success_probabilities

        
    def generate_prob_array(self):
        return self.success_probabilities

    def best_expectation(self):
        success_prob = sorted(self.success_probabilities)[::-1]
        expected_slots = 0
        none_avail = 1

        for i in range(len(success_prob)):
            none_avail = none_avail * (1 - success_prob[i])
            ival = i + 1
            for j in range(i):
                ival = ival * (1 - success_prob[j])
            ival = ival * success_prob[i]
            expected_slots = expected_slots + ival

        expected_slots = expected_slots + len(success_prob) * none_avail
        #print(expected_slots)
        return expected_slots*(self.exploit_rounds+self.explore_rounds)
    
    def update_params(self):
        indices_descending = np.argsort(self.means_array)[::-1]
        self.sorted_prob_exploration = self.success_probabilities[indices_descending]
        #print(self.sorted_prob_exploration)
        #return self.sorted_prob_exploration
    
    def explore(self):
        for i in range(self.explore_rounds):
            self.cost = self.cost + self.size
            binary_array = np.random.uniform(0,1,self.size) < self.success_probabilities
            for k in range(len(binary_array)):
                if binary_array[k] == 1:
                    self.means_array[k] = self.means_array[k] + 1
        #print(self.means_array)
        

    def exploit(self):
       for il in range(self.exploit_rounds):
            
            binary_array = np.random.uniform(0,1,self.size) < self.sorted_prob_exploration
            #print(binary_array)
            no_pick = np.all(binary_array == 0)

            if no_pick:
                self.cost = self.cost + self.size
                #print(':(')
            else:
                self.cost = self.cost + np.argmax(binary_array) + 1
                #print(np.argmax(binary_array))
       return self.cost
    

    @staticmethod
    def experiment(initial_percent, final_percent,total_rounds, step, experiments, primaries):
        
        avg_best_cost = np.zeros((final_percent-initial_percent-1)//step + 1,dtype=float)
        avg_cost = np.zeros((final_percent-initial_percent-1)//step + 1,dtype=float)
        avg_regret = np.zeros((final_percent-initial_percent-1)//step + 1,dtype=float)

        for a in range(experiments):
            bandit = MultiArmedBandit(primaries,0,0)
            #print(bandit.success_probabilities)
            cost = np.array([])
            best_cost = np.array([])
            regret = np.array([])
            for i in range(initial_percent,final_percent,step):
                
                
                
                bandit.explore_rounds = ((total_rounds*i)//100)
                bandit.exploit_rounds = total_rounds - bandit.explore_rounds
                bandit.means_array = np.zeros(bandit.size, dtype=float)
                bandit.cost=0
                bandit.sorted_prob_exploration = bandit.success_probabilities
            
                be = bandit.best_expectation()
                best_cost = np.append(best_cost,be)
                bandit.explore()
                bandit.update_params()
                exp = bandit.exploit()
                cost = np.append(cost,exp)
                regret = np.append(regret,exp - be)
                
            avg_regret = avg_regret + regret / experiments
            avg_cost = avg_cost + cost/experiments
            avg_best_cost = avg_best_cost + best_cost/experiments
        #print(avg_regret) 
        #print( avg_cost)
        #print( avg_best_cost)
        percent_array = np.arange(initial_percent,final_percent,step)
        return np.column_stack((avg_regret,percent_array))


#bandit = MultiArmedBandit(10)
#print(MultiArmedBandit.experiment(0, 1000, 1, 3))
total_rounds = 1000
primaries = 20
total_experiments = 100
steps = 1
initial_percent = 0      #these percentages indicate percent of the total rounds given for exploration
final_percent = 10
initial_primary = 10
final_primary = 100
step_for_primary = 10
for i in range(initial_primary,final_primary+1,step_for_primary):
    print(i)
    scores = MultiArmedBandit.experiment(initial_percent,final_percent+1,total_rounds,steps,total_experiments,i)
    x_val = scores[:,1]
    y_val = scores[:,0]
    plt.plot(x_val,y_val,label=f'{i}')


plt.legend()        
plt.show()       #regret vs percent for exploration plot for different primaries.


