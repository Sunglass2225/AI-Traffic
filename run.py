from asyncio.windows_events import NULL
import pandas as pd
import numpy as np
from Network import Network
from Controller import MaxPressureController
from Controller import dqnController
import traci
import os
import sys
import json
from DQN_Agent import DQNAgent
from data_logger import Data_Logger

if __name__ == "__main__":

    controller_type = "max_pressure"

    # LOAD SUMO STUFF
    cfgfilename = "3_16RUNNER.sumo.cfg"
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filepath = os.path.join(dir_path,"network",cfgfilename)
    print(filepath)

    #sumoCmd = ["sumo", "-c", filepath]
    sumoCmd = ["sumo-gui", "-c", filepath]  # if you want to see the simulation

    # initialize the network object and controller object
    tracilabel = "sim1"
    traci.start(sumoCmd, label=tracilabel)
    conn = traci.getConnection(tracilabel)

    
    controller = None
    agent = None
    if controller_type == "max_pressure":
        controller = MaxPressureController()
    else:
        controller = dqnController()
        agent = DQNAgent()
        try:
            agent.load('DQN_control_9.h5')
            print('Agent_loaded')
        except:
            print('No models found')

    runID = 'max_pressure_one_intersection'
    logger = Data_Logger(runID)
    network = Network(filepath, conn, agent)        
    step = 0
    action = 0

    while conn.simulation.getMinExpectedNumber() > 0:
        conn.simulationStep()
        if step > 1 and step % 10 == 0:

            # get current state

            intersections = list(network.network.keys())
            print("intersections" + str(intersections))
            print("in step " + str(step))

            if controller_type == "max_pressure":
                intersection = intersections[0]
                state = network.getState(conn,intersection)
                geometry = network.getGeometry(intersection)
                print(state)

                # get maxpressure controller
                control = controller.getController(geometry,state)
                print("   " + intersection + " light list : " + str(control))
                # update the state of the network
                network.applyControl(control,conn,intersection)      
                
                #########write_state_to_file(state)   
                #metrics = updateMetrics(conn,metrics,state,geometry)
            
            if controller_type == "idqn_Controller":
                geometry = network.DQNgeometry
                DQN_state = network.DQN_getstate(conn, action)[0]
                state = network.DQN_getstate(conn, action)[1]

                result = controller.getController(DQN_state, geometry, conn, agent)
                control = result[0]
                action = result[1]
                print("controller: ", control)
                print("action: ", action)

                
                network.applyControl(control,conn, geometry["intersections"][0])

            # write_state_to_file(state)
            logger.updateLane(step, conn, network.allLaneId)
            logger.updateVeh(step, conn, state)

                    
           
        step += 1


   

    traci.close(False)

