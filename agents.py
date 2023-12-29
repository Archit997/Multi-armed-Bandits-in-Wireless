import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
class ArmedBanditsEnv():
    """
    num_expt -> number of experiments 
    num_slots -> available slots that can transmit data based on availablility
    p_values -> num_expts x num_slots matrix containing p-values for availability of slot
    action -> num_expts x num_slots array denoting the order of checking slots for availability for each expt
    """
    
    def __init__(self, p_values):
        assert len(p_values.shape) == 2
        
        self.num_slots = p_values.shape[1]
        self.num_expts = p_values.shape[0]
        self.state = np.zeros((self.num_expts,self.num_slots))
        
        self.p_values = p_values

    def generate_state(self):
        return np.random.binomial(n=1,p=self.p_values)
    
        
    def step(self, action_type, action, explore_type):
        
        # Sample from the specified slot using it's bernoulli distribution
        assert (action.shape == (self.num_expts,self.num_slots))
        
        sampled_state = np.random.binomial(n=1, p=self.p_values)

        self.state = sampled_state

        cost = np.zeros((self.num_expts, 1))

        for j in range(self.num_expts):
            # Get the relevant actions and their indices for the current experiment
            if action_type[j]==0 and explore_type[j]==1:
                cost[j]=self.num_slots
            else:
                actions = np.array(action[j]) - 1  # Adjust for zero-based indexing
                relevant_states = sampled_state[j, actions]
                # Find the index of the first occurrence of 1, if any
                indices = np.where(relevant_states == 1)[0]
                if indices.size > 0:
                    first_one_index = indices[0]
                else:
                    first_one_index = len(actions)-1
                # Calculate the cost
                cost[j] = first_one_index+1
        
        # Return a constant state of 0. Our environment has no terminal state
        observation, done, info = 0, False, dict()
        assert (cost.shape[1]==1)
        return cost,self.state, done, info
    
    def reset(self):
        return 0
        
    def render(self, mode='human', close=False):
        pass
    
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np.random(seed)
        return [seed]
    
    def close(self):
        pass
    
    
class ArmedBanditsBernoulli(ArmedBanditsEnv):
    def __init__(self, num_expts=1, num_slots=5):
        self.p_values = np.random.uniform(0, 1, (num_expts, num_slots))
        
        ArmedBanditsEnv.__init__(self, self.p_values)


class General():
    def __init__(self):
        return 0
    
    def inc_p(self,prev_p, new_val, n):
        return (prev_p*(n-1) + new_val)/n
    
    def sort_with_noise(self,row):
        # Identifying unique values and their counts
        unique, counts = np.unique(row, return_counts=True)

        # Only add noise to elements where there are ties
        for value, count in zip(unique, counts):
            if count > 1:
                noise = np.random.normal(0, 1e-6, count)  # Small noise
                indices = row == value
                row[indices] += noise

        # Sort the indices after adding noise
        sorted_indices = np.argsort(-row) + 1
        return sorted_indices
    
    def greedy_order(self,estimates):
        """
        Takes in an array of estimates of num_expts x num_slots and returns the order
        of slots with the decreasing estimated p_value for each row. 
        Breaks ties randomly by introducing a small random noise.
        """
        if estimates.ndim == 1:
            sorted_indices = self.sort_with_noise(estimates)
        else:
            sorted_indices = np.apply_along_axis(self.sort_with_noise, 1, estimates)
        return sorted_indices
    
    def random_order(self,row):
        perm_index = np.random.permutation(len(row))

        return perm_index

    def random_ordering(self,estimates):
        """
        Takes in an array of estimates of num_expts x num_slots and returns the order
        of slots with the decreasing estimated p_value for each row.
        Breaks ties randomly by introducing a small random noise.
        """
        if estimates.ndim ==1:
            perm_index =self.random_order(estimates)

        else:
            perm_index = np.apply_along_axis(self.random_order, 1, estimates)
        return perm_index
    

class OptimalAgent(General):
    def __init__(self, p_values):
        super().__init__()
        # Store the epsilon value
        
        
        self.num_slots = p_values.shape[1]
        self.num_expts = p_values.shape[0]
        self.estimates = p_values.astype(np.float64)
        
        
    
    def get_action(self):
        # We need to redefine this function so that it takes an exploratory action with epsilon probability
        
        # One hot encoding: 0 if exploratory, 1 otherwise
        action_type = np.tile([1],(self.num_expts,1))
        explore_type = np.tile([1],(self.num_expts,1)) 
        # Generate both types of actions for every experiment
        
        # Use the one hot encoding to mask the actions for each experiment
        action = self.greedy_order(self.estimates)
               
        return action_type,  action, explore_type
    
    def update_estimates(self, cost, state, action_type, action,explore_type):

        return 0
    
class EpsilonGreedyAgent(General):
    def __init__(self, estimates, epsilon, delta):

        super().__init__()

        # Store the epsilon value
        assert epsilon >= 0 and epsilon <= 1
        assert delta >= 0 and delta <= 1
        assert len(estimates.shape) == 2
        
        self.num_slots = estimates.shape[1]
        self.num_expts = estimates.shape[0]
        self.estimates = estimates.astype(np.float64)
        self.action_count = np.zeros(estimates.shape)
        self.epsilon = epsilon
        self.delta = delta
        
    
    def get_action(self):

        #epsilon is 1 for 100% greedy, delta is 0 for 100% full exploration
        action_type = (np.random.random_sample(self.num_expts) > self.epsilon).astype(int).reshape(-1,1)
        explore_type = (np.random.random_sample(self.num_expts) > self.delta).astype(int).reshape(-1,1)
        # Generate both types of actions for every experiment
        explore_full = np.tile(np.arange(1, self.num_slots + 1), (self.num_expts, 1))
        greedy_action = self.greedy_order(self.estimates)
        random_action = self.random_ordering(self.estimates)
        exploratory_action = explore_full*explore_type + random_action*(1-explore_type)
        # Use the one hot encoding to mask the actions for each experiment
        action = greedy_action * action_type + exploratory_action * (1 - action_type)

        # Action type = 1 => Greedy
        # Action type = 0 => explore
        # Explore type = 1 => explore full
        # Explore type = 0 => explore partially
        return action_type,action, explore_type
    
    def update_estimates(self, cost, state, action_type, action,explore_type):
        for j in range(self.num_expts):
            if action_type[j]==0 and explore_type[j]==1:
                # Increment action_count for all slots
                self.action_count[j] += 1
        
                # Apply the vectorized inc_p function to the entire row
                n = self.action_count[j]
                self.estimates[j] = self.inc_p(self.estimates[j], state[j], n)
            else:
                #print(self.estimates)
                num_expts, num_slots = action.shape

            
                c = int(cost[j][0])-1

                if c>0:
                    changed = action[j][:c]-1
                    
                    # Update the estimates and counts for the changed actions
                    self.action_count[j, changed] += 1
                    #print(self.action_count)
                    self.estimates[j, changed] = self.inc_p(self.estimates[j, changed], 0, self.action_count[j, changed])
                    #print(self.estimates)
                    

                # Update the estimates and counts for the not-changed action, if it exists
                slot = action[j][c]-1

                if c == num_slots-1:
                    self.action_count[j, slot] += 1
                    self.estimates[j, slot] = self.inc_p(self.estimates[j, slot], state[j,slot], self.action_count[j, slot])
                else :
                    self.action_count[j, slot] += 1
                    #print(self.action_count)
                    #print(self.action_count)
                    self.estimates[j, slot] = self.inc_p(self.estimates[j, slot], 1, self.action_count[j, slot])
                
             
        
class UCBAgent(General):
    def __init__(self, estimates,radius, epsilon ,steps, delta):

        super().__init__()
        
        # Store the epsilon value
        assert epsilon >= 0 and epsilon <= 1
        assert len(estimates.shape) == 2
        assert delta >= 0 and delta <= 1
        self.num_slots = estimates.shape[1]
        self.num_expts = estimates.shape[0]
        self.estimates = estimates.astype(np.float64)
        self.radius = radius.astype(np.float64)
        self.action_count = np.zeros(estimates.shape)
        self.epsilon = epsilon
        self.total_steps = steps
        self.delta = delta
        
    
    def get_action(self):
        # We need to redefine this function so that it takes an exploratory action with epsilon probability
        
        # One hot encoding: 0 if exploratory, 1 otherwise
        action_type = (np.random.random_sample(self.num_expts) > self.epsilon).astype(int).reshape(-1,1)
        explore_type = (np.random.random_sample(self.num_expts) > self.delta).astype(int).reshape(-1,1)
        # Generate both types of actions for every experiment
        explore_full = np.tile(np.arange(1, self.num_slots + 1), (self.num_expts, 1))
        greedy_action = self.greedy_order(self.estimates)
        random_action = self.random_ordering(self.estimates)
        exploratory_action = explore_full*explore_type + random_action*(1-explore_type)
        # Use the one hot encoding to mask the actions for each experiment
        action = greedy_action * action_type + exploratory_action * (1 - action_type)
        
        return action_type,  action, explore_type
    
    def update_estimates(self, cost, state, action_type, action, explore_type):
        for j in range(self.num_expts):
            if action_type[j]==0 and explore_type[j]==1:
                # Increment action_count for all slots
                self.action_count[j] += 1
        
                # Apply the vectorized inc_p function to the entire row
                n = self.action_count[j]
                self.estimates[j] = self.inc_p(self.estimates[j], state[j], n)
            else:

                num_expts, num_slots = action.shape

                c = int(cost[j][0])-1

                if c>0:
                    changed = action[j][:c]-1

                    # Update the estimates and counts for the changed actions
                    self.action_count[j, changed] += 1

                    self.estimates[j, changed] = self.inc_p(self.estimates[j, changed], 0, self.action_count[j, changed])

                # Update the estimates and counts for the not-changed action, if it exists
                slot = action[j][c]-1

                if c == num_slots-1:
                    self.action_count[j, slot] += 1
                    self.estimates[j, slot] = self.inc_p(self.estimates[j, slot], state[j,slot], self.action_count[j, slot])
                else :
                    self.action_count[j, slot] += 1
                    #print(self.action_count)
                    #print(self.action_count)
                    self.estimates[j, slot] = self.inc_p(self.estimates[j, slot], 1, self.action_count[j, slot])
                    
        zero_mask = self.action_count == 0
        non_zero_mask = ~zero_mask

        # Calculate radius for slots with non-zero action_count
        self.radius[non_zero_mask] = np.sqrt(2 * np.log(self.total_steps) / self.action_count[non_zero_mask])

        # Set radius to 0 for slots with zero action_count
        self.radius[zero_mask] = 0 
        

                
                
            
class TSAgent(General):
    def __init__(self, alpha, beta):
        super().__init__()
        # Store the epsilon value
        alpha = alpha.astype(np.float64)
        beta = beta.astype(np.float64)
        
        assert len(alpha.shape) == 2
        assert len(beta.shape) == 2

        assert alpha.all()>0 and beta.all()>0

        self.num_slots = alpha.shape[1]
        self.num_expts = beta.shape[0]
        self.alpha = alpha.astype(np.float64)
        self.beta = beta.astype(np.float64)
        self.estimates = np.zeros(alpha.shape)
        self.action_count = np.zeros(alpha.shape)
        
    
    def get_action(self):

        sampled_estimates = np.random.beta(self.alpha,self.beta,self.alpha.shape)
        action = self.greedy_order(sampled_estimates)
        self.estimates = sampled_estimates

        return np.tile(1,(self.num_expts,1)),action ,np.tile(0,(self.num_expts,1))
    
    def update_estimates(self, cost, state, action_type, action,explore_type):
        for j in range(self.num_expts):
            #print(self.estimates)
            num_expts, num_slots = action.shape
        
            c = int(cost[j][0])-1
            if c>0:
                changed = action[j][:c]-1
                
                # Update the estimates and counts for the changed actions
                self.action_count[j, changed] += 1
                #print(self.action_count)
                self.beta[j, changed] += 1
                #print(self.estimates)
                
            # Update the estimates and counts for the not-changed action, if it exists
            slot = action[j][c]-1
            if c == num_slots-1:
                self.action_count[j, slot] += 1
                if state[j][slot]==1:
                    self.alpha[j, slot] += 1
                else:
                    self.beta[j,slot] +=1
            else :
                self.action_count[j, slot] += 1
                #print(self.action_count)
                #print(self.action_count)
                self.alpha[j, slot] += 1

            assert self.alpha.all()>0 and self.beta.all()>0
                
 class DS(General):
    def __init__(self,mean,variance,samples,alpha, beta, prop_factor):
        super().__init__()
        self.mean = mean.astype(np.float64)
        self.variance = variance.astype(np.float64)
        self.slots  = mean.shape[1]
        self.samples = samples
        self.alpha = alpha
        self.beta = beta
        self.num_expts = mean.shape[0]
        self.estimates = np.zeros(alpha.shape)
        self.action_count = np.zeros(alpha.shape)
        self.prop_factor = prop_factor
    def get_action(self):
        
        samples_matrix = np.zeros((self.num_expts,self.samples,self.slots))
        
        for j in range(self.num_expts):
         for i in range(self.slots):
            
            samples_matrix[j,:,i] = np.random.beta(self.alpha[j][i],self.beta[j][i],self.samples)
        self.estimates = samples_matrix 
          
        for j in range(self.num_expts):
            freq_array = np.zeros(self.slots)
            for i in range(self.samples):
                freq_array[np.argmax(samples_matrix[j][i])]+=1
              
            self.mean[j] = freq_array/self.samples
            self.variance[j] = np.multiply(np.square(1 - self.mean[j])/self.samples,freq_array) + np.multiply((np.square(self.mean[j])/self.samples),self.samples-freq_array)
        arm_freq = np.zeros(self.num_expts)
        
        for i in range(self.num_expts):
            max_prob= np.max(self.mean[i])
            pfa = 0
            for j in range(self.slots):
                if self.mean[i][j]!=max_prob:
                  if self.variance[i][j] !=0:
                   val = (max_prob-self.mean[i][j])/math.sqrt(self.variance[i][j])
                   
                   pfa+=((1-norm.cdf(val,0,1))/(self.slots-1))
            arm_freq[i] = pfa       
        
        
        arm_freq=np.log(1/arm_freq)
        
        arm_freq = self.slots*self.prop_factor*arm_freq
        
        slot_freq = np.zeros((self.num_expts,self.slots))
        
        for j in range(self.num_expts):
             #print(round(arm_freq[j]))
            if arm_freq[j]!= float('inf'): 
             slot_freq[j] = draw_samples(self.mean[j],round(arm_freq[j]))
            else:
             slot_freq[j] = draw_samples(self.mean[j],round(self.slots*self.prop_factor*10))    
        action = self.greedy_order(slot_freq) 
            
        #print(slot_freq)
        #print(action)
        return np.tile(1,(self.num_expts,1)),action ,np.tile(0,(self.num_expts,1))
    
    def update_estimates(self, cost, state, action_type, action,explore_type):
        for j in range(self.num_expts):
            #print(self.estimates)
            num_expts, num_slots = action.shape
        
            c = int(cost[j][0])-1
            if c>0:
                changed = action[j][:c]-1
                
                # Update the estimates and counts for the changed actions
                self.action_count[j, changed] += 1
                #print(self.action_count)
                self.beta[j, changed] += 1
                #print(self.estimates)
                
            # Update the estimates and counts for the not-changed action, if it exists
            slot = action[j][c]-1
            if c == num_slots-1:
                self.action_count[j, slot] += 1
                if state[j][slot]==1:
                    self.alpha[j, slot] += 1
                else:
                    self.beta[j,slot] +=1
            else :
                self.action_count[j, slot] += 1
                #print(self.action_count)
                #print(self.action_count)
                self.alpha[j, slot] += 1

            assert self.alpha.all()>0 and self.beta.all()>0            
        
class experiment():
    def __init__(self,agents,env,num_expts,num_slots,num_steps):
        self.agents = agents
        self.env = env
        self.num_expts = num_expts
        self.num_slots = num_slots
        self.num_steps = num_steps
        self.costs = {}
        self.estimates={}
        self.regret={}
        #self.give_Cost()
        #self.test_cost()
        self.agents["Optimal"] = OptimalAgent(self.env.p_values)
        for i,name in enumerate(agents.keys()):
            self.costs[name] = np.zeros((self.num_expts,self.num_steps+1))


    def give_Cost(self,num_expts,num_slots,sampled_state,action_type,action, explore_type):
        cost = np.zeros((num_expts, 1))

        for j in range(num_expts):
            # Get the relevant actions and their indices for the current experiment
            if action_type[j]==0 and explore_type[j]==1:
                cost[j]=num_slots
            else:
                actions = np.array(action[j]) - 1  # Adjust for zero-based indexing
                relevant_states = sampled_state[j, actions]
                # Find the index of the first occurrence of 1, if any
                indices = np.where(relevant_states == 1)[0]
                if indices.size > 0:
                    first_one_index = indices[0]
                else:
                    first_one_index = len(actions)-1
                # Calculate the cost
                cost[j] = first_one_index+1
        return cost 

    def test_cost(self,costs,step,agents,env):
        state = env.generate_state()
        
        num_expts = state.shape[0]
        num_slots = state.shape[1]
        
        for i,name in enumerate(agents.keys()):
            
            action_type,action,explore_type = agents[name].get_action()
            
            cost = self.give_Cost(num_expts,num_slots,state,action_type,action,explore_type)
            agents[name].update_estimates(cost,state,action_type,action,explore_type)
            costs[name][:,step+1] = costs[name][:,step] + cost.flatten()

        return costs,agents
    
    def run(self):
        
        for i in tqdm(range(self.num_steps)):
            self.costs,self.agents = self.test_cost(self.costs,i,self.agents,self.env)

        for i,name in enumerate(self.agents.keys()):
            self.estimates[name] = (self.agents[name].estimates).mean(axis=0)
            self.regret[name] = (self.costs[name] - self.costs["Optimal"]).mean(axis=0)

    
    def plot(self):
        plt.figure(figsize=(12,6), dpi=80, facecolor='w', edgecolor='k')
        
        for i,name in enumerate(self.agents.keys()):
            
            plt.plot(self.regret[name])

        plt.legend(self.agents.keys())
        plt.title(f"Testing algorithms on {self.num_steps} rounds over {self.num_slots} bandits")
        plt.ylabel(f"Avg Regret over {self.num_expts} realizations")
        plt.xlabel("Steps")
        plt.show()        

    

    


