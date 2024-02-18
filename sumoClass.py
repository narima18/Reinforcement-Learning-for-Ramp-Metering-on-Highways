from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import optparse
import random
import traci
import numpy as np
import math

import configparser
from sumolib import checkBinary


class SumoClass:
    
    def __init__(self,):
        print("preparation to launch simulation .....")
    
    def get_options(self,gui, sumocfg_file_name, max_steps):
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")
        # setting the cmd mode or the visual mode    
        if gui == False:
            sumoBinary = checkBinary('sumo')
        else:
            sumoBinary = checkBinary('sumo-gui')
 
        # setting the cmd command to run sumo at simulation time
        sumo_cmd = [sumoBinary, "-c",sumocfg_file_name, "--no-step-log", "true", "--waiting-time-memory", str(max_steps)]

        return sumo_cmd
    
    
    
    def generate_routefile(self, seed):
        """
        Generation of the route of every car for one episode
        """
        np.random.seed(seed)  # make tests reproducible

        # the generation of cars is distributed according to a weibull distribution
        timings = np.random.weibull(4, 1800)
        timings = np.sort(timings)

        # reshape the distribution to fit the interval 0:max_steps
        car_gen_steps = []
        min_old = math.floor(timings[1])
        max_old = math.ceil(timings[-1])
        min_new = 0
        max_new = 5500
        for value in timings:
            car_gen_steps = np.append(car_gen_steps, ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)

        car_gen_steps = np.rint(car_gen_steps)  # round every value to int -> effective steps when a car will be generated

        # produce the file for cars generation, one car per line
        with open("route.rou.xml", "w") as routes:
            print("""<routes> 
                  <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />

            <route id="E8" edges="E8 E10"/>
            <route id="E9" edges="E9 E10"/>
""", file=routes)

            for car_counter, step in enumerate(car_gen_steps):
                straight_or_turn = np.random.uniform()
                if straight_or_turn < 0.75:  # choose direction: straight or turn - 75% of times the car goes straight
                    print('    <vehicle id="E8%i" type="standard_car" route="E8" depart="%s" departLane="random" departSpeed="10" arrivalLane="random" />' % (car_counter, step), file=routes)
                else:  # car that turn -25% of the time the car turns
                    print('    <vehicle id="E9%i" type="standard_car" route="E9" depart="%s" departLane="random" departSpeed="10" arrivalLane="random" />' % (car_counter, step), file=routes)

            print("</routes>", file=routes)


    def get_state(self):
        
        state = np.zeros(20)
        car_list = traci.vehicle.getIDList()
        
        for car_id in car_list:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)
            lane_pos = 750 - lane_pos  # inversion of lane pos, so if the car is close to the traffic light -> lane_pos = 0 --- 750 = max len of a road
            lane_cell = 0
            if lane_id == "E8_0" or lane_id == "E8_1" or lane_id == "E8_2":
                lane_group = 0
                if lane_pos < 6:
                    lane_cell = 0
                elif lane_pos < 12:
                    lane_cell = 1
                elif lane_pos < 18:
                    lane_cell = 2
                elif lane_pos < 24:
                    lane_cell = 3
                elif lane_pos < 35:
                    lane_cell = 4
                elif lane_pos < 60:
                    lane_cell = 5
                elif lane_pos < 120:
                    lane_cell = 6
                elif lane_pos < 200:
                    lane_cell = 7
                elif lane_pos < 450:
                    lane_cell = 8
                elif lane_pos <=600 :
                    lane_cell = 9
            elif lane_id == "E9_0":
                lane_group = 1
                if lane_pos < 5:
                    lane_cell = 0
                elif lane_pos < 15:
                    lane_cell = 1
                elif lane_pos < 24:
                    lane_cell = 2
                elif lane_pos < 32:
                    lane_cell = 3
                elif lane_pos < 45:
                    lane_cell = 4
                elif lane_pos < 60:
                    lane_cell = 5
                elif lane_pos < 80:
                    lane_cell = 6
                elif lane_pos < 110:
                    lane_cell = 7
                elif lane_pos < 160:
                    lane_cell = 8
                elif lane_pos <=200 :
                    lane_cell = 9
            else:
                lane_group = -1
            # distance in meters from the traffic light -> mapping into cells
            print(lane_group)
            cell = lane_cell
            if lane_group == 0:
                car_position = cell
                valid_car = True
            elif lane_group ==1:
                car_position = int(str(lane_group) + str(cell))  # composition of the two postion ID to create a number in interval 0-79
                valid_car = True
           
            else:
                valid_car = False  # flag for not detecting cars crossing the intersection or driving away from it
            
            if valid_car:
                state[car_position] = 1  # write the position of the car car_id in the state array in the form of "cell occupied"

        return state
    


