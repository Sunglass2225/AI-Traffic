from asyncio.windows_events import NULL
import pandas as pd
import numpy as np
from Network import Network
from Controller import MaxPressureController
from Controller import IDQNcontroller
import traci
import os
import sys
import json
from DQN_Agent import DQNAgent
from data_logger import Data_Logger

if __name__ == "__main__":

    controller_type = "idqn_Controller"

    # LOAD SUMO STUFF
    cfgfilename = "SUMO_Network.sumo.cfg"
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filepath = os.path.join(dir_path,"network",cfgfilename)
    print(filepath)

    #sumoCmd = ["sumo", "-c", filepath]
    sumoCmd = ["sumo-gui", "-c", filepath]  # if you want to see the simulation

    runID = 'BBB'
    #create data logger, pass in runID
    logger = Data_Logger(runID)

    # initialize the network object and controller object
    tracilabel = "sim1"
    traci.start(sumoCmd, label=tracilabel)
    conn = traci.getConnection(tracilabel)

    network = Network(filepath, conn)
    controller = None
    agent = None
    if controller_type == "max_pressure":
        controller = MaxPressureController()
    else:
        controller = IDQNcontroller()
        agent = DQNAgent()
        try:
            agent.load('DQN_control_2.h5')
            print('Agent_loaded')
        except:
            print('No models found')

    step = 0
    action = 0
    reward_list = [0, 0, 0]
    state_list = [0, 0, 0]
    action_list = [0, 0, 0]

    while conn.simulation.getMinExpectedNumber() > 0:
        conn.simulationStep()
        if step > 1 and step % 10 == 0:

            # get current state

            intersections = list(network.network.keys())
            print("intersections" + str(intersections))
            print("in step " + str(step))

            if controller_type == "max_pressure":
                for i in range(len(intersections)):
                    intersection = intersections[i]
                    state = network.getState(conn,intersection)
                    geometry = network.getGeometry(intersection)

                    # get maxpressure controller
                    control = controller.getController(geometry,state)
                    print("   " + intersection + " light list : " + str(control))
                    # update the state of the network
                    network.applyControl(control,conn,intersection)      
                
                    #########write_state_to_file(state)   
                    #metrics = updateMetrics(conn,metrics,state,geometry)
            
            if controller_type == "idqn_Controller":
                print("Start")

                for i in range(len(intersections)):         
                    intersection = intersections[i]
                    geometry = network.getGeometry(intersection)
                    action  = action_list[i]

                    state = network.IDQN_getstate(conn, intersection, action)
                    state_metric = network.getState(conn, intersection)

                    result = controller.getController(state[0], geometry, agent)
                    control = result[0]
                    action = result[1]
                    action_list[i] = action
                    print("   " + intersection + " light list : " + str(control))
                    print("   " + intersection + " Action : " + str(action + 1))
                    # update the state of the network
                    network.applyControl(control,conn,intersection)   

                logger.updateLane(step, conn, network.allLaneId)
                logger.updateVeh(step, conn, state_metric)
           
        step += 1


   

    traci.close(False)

