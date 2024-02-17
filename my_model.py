from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import random
import numpy as np
from collections import deque
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # kill warning about tensorflow
import tensorflow as tf
from sumolib import checkBinary
import traci

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

from sumoClass import SumoClass

class TLSClass:
    
    def __init__(self):
        self.gamma = 0.95
        self.learning_rate = 0.001
        self.memory = []
        self.action_size = 1
        self._input_dim = 20
        self._output_dim = 1
        self.model = self._build_model(6,400)
        self._size_max = 50000
        self._size_min = 600
        self.batch_size = 32
        self._num_states = 20
        self._num_actions = 2
        self._gamma = 0.85

    
    def _build_model(self, num_layers, width):
        """
        Build and compile a fully connected deep neural network
        """
        inputs = keras.Input(shape=(self._input_dim,))
        x = layers.Dense(width, activation='relu')(inputs)
        for _ in range(num_layers):
            x = layers.Dropout(0.1)(x)
            x = layers.Dense(width, activation='relu')(x)
        outputs = layers.Dense(self._output_dim, activation='linear')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='my_model')
        model.compile(loss=losses.mean_squared_error, optimizer=Adam(lr=0.001))
        return model
    

    def remember(self, sample):
        self.memory.append(sample)
        if len(self.memory) > self._size_max:
            self.memory.pop(0)  # if the length is greater than the size of memory, remove the oldest element
    
    def get_samples(self, n):
        """
        Get n samples randomly from the memory
        """
        if len(self.memory) < self._size_min:
            return []

        if n >len(self.memory):
            return random.sample(self.memory, len(self.memory))  # get all the samples
        else:
            return random.sample(self.memory, n)  # get "batch size" number of samples
       
    
    def act(self, state,epsilon):
        if np.random.rand() <= epsilon:
            return random.randrange(self.action_size)
        state = np.reshape(state, [1, self._input_dim])
        act_values = self.model.predict(state)
        
        return np.argmax(act_values)

    

    def load(self, name):
        self.model.load_weights(name)
    
    def save(self, name):
        self.model.save_weights(name)

    def _collect_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["E8", "E9"]
        waiting_times = {}
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                waiting_times[car_id] = wait_time
            else:
                if car_id in waiting_times: # a car that was tracked has cleared the intersection
                    del waiting_times[car_id] 
        total_waiting_time = sum(waiting_times.values())
        return total_waiting_time
    
    def _simulate(self, steps_todo,step,sum_queue_length, sum_waiting_time):
        """
        Execute steps in sumo while gathering statistics
        """
        if (step + steps_todo) >=5500:  # do not do more steps than the maximum allowed number of steps
            steps_todo =5500 -step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            step += 1 # update the step counter
            steps_todo -= 1
            halt_N = traci.edge.getLastStepHaltingNumber("E8")
            halt_S = traci.edge.getLastStepHaltingNumber("E9")
            queue_length = halt_N + halt_S
            sum_queue_length += queue_length
            sum_waiting_time += queue_length # 1 step while wating in queue means 1 second waited, for each car, therefore queue_lenght == waited_seconds
        
        return sum_queue_length, sum_waiting_time,step
    
    def _replay(self):
        """
        Retrieve a group of samples from the memory and for each of them update the learning equation, then train
        """
        batch = self.get_samples(self.batch_size)

        if len(batch) > 0:  # if the memory is full enough
            states = np.array([val[0] for val in batch])  # extract states from the batch
            next_states = np.array([val[3] for val in batch])  # extract next states from the batch

            # prediction
            q_s_a = self.model.predict(states)  # predict Q(state), for every sample
            q_s_a_d = self.model.predict(next_states)  # predict Q(next_state), for every sample

            # setup training arrays
            x = np.zeros((len(batch), self._num_states))
            y = np.zeros((len(batch), self._num_actions))

            for i, b in enumerate(batch):
                state, action, reward, _ = b[0], b[1], b[2], b[3]  # extract data from one sample
                current_q = q_s_a[i]  # get the Q(state) predicted before
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])  # update Q(state, action)
                x[i] = state
                y[i] = current_q  # Q(state) that includes the updated action value

            self.model.fit(x, y, epochs=1, verbose=0)  # train the NN


if __name__ == "__main__":
    
    
    sumoObject = SumoClass()
    options = sumoObject.get_options(False,'config.sumocfg',5400)

    
    # To be removed
    #sumoBinary = checkBinary('sumo')
    

    
    num_episodes = 100
    tlsObject = TLSClass()
    
   

    # Pre-load the weights
    '''try:
        tlsObject.load('trained_kernel_initialiser.h5')
    except:
        print("No models found to instantiate kernel")
    '''

    reward_store = []
    cumulative_wait_store = []
    avg_queue_length_store = []

    for episode in range(num_episodes):
        print(episode)
        epsilon = 1.0 - (episode /num_episodes)
        stepz = 0
        old_total_wait = 0
        old_state = -1
        old_action = -1
        sum_neg_reward=0
        sum_queue_length, sum_waiting_time = (0,0)
        sumoObject.generate_routefile(seed=episode)
        traci.start(options)
        while  stepz < 5500:   # as long as vehicles are there
        
            currentState = sumoObject.get_state()
            current_total_wait = tlsObject._collect_waiting_times()
            reward = old_total_wait - current_total_wait
            if stepz >0 : 
                tlsObject.remember((old_state, action, reward, currentState))
            
            action = tlsObject.act(currentState,epsilon)

             # if the chosen phase is different from the last phase, activate the yellow phase
            if stepz != 0 and old_action != action:
                traci.trafficlight.setPhase("J15_1",old_action*2+1)
                sum_queue_length, sum_waiting_time,stepz = tlsObject._simulate(6,stepz,sum_queue_length, sum_waiting_time)


            # execute the phase selected before
            traci.trafficlight.setPhase("J15_1",action*2)
            sum_queue_length, sum_waiting_time ,stepz= tlsObject._simulate(39,stepz,sum_queue_length, sum_waiting_time)

            # saving variables for later & accumulate reward
            old_state = currentState
            old_action = action
            old_total_wait = current_total_wait

            # saving only the meaningful reward to better see if the agent is behaving correctly
            if reward < 0:
                sum_neg_reward += reward

         
        reward_store.append(sum_neg_reward) 
        cumulative_wait_store.append(sum_waiting_time) 
        avg_queue_length_store.append(sum_queue_length /5500)  
        traci.close(wait=False)

        for _ in range(800):
            tlsObject._replay()
        

    
    # Saving weights
    tlsObject.save('trained_kernel_initialiser_2.h5')

    sys.stdout.flush()