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

if __name__ == "__main__":

    controller_type = "dqn_Controller"

    # LOAD SUMO STUFF
    cfgfilename = "SUMO_Network.sumo.cfg"
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filepath = os.path.join(dir_path,"network",cfgfilename)
    print(filepath)

    sumoCmd = ["sumo", "-c", filepath]
    #sumoCmd = ["sumo-gui", "-c", filepath]  # if you want to see the simulation

    # initialize the network object and controller object
    tracilabel = "sim1"
    traci.start(sumoCmd, label=tracilabel)
    conn = traci.getConnection(tracilabel)

    network = Network(filepath, conn)
    controller = "max_pressure"
    agent = None
    if controller_type == "max_pressure":
        controller = MaxPressureController()
    else:
        controller = dqnController()
        agent = DQNAgent()
        try:
            agent.load('DQN_control_0.h5')
            print('Agent_loaded')
        except:
            print('No models found')

    step = 0
    action = 0

    while conn.simulation.getMinExpectedNumber() > 0:
        conn.simulationStep()
        if step > 1 and step % 30 == 0:

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
            
            if controller_type == "dqn_Controller":
                geometry = network.DQNgeometry
                state = network.DQN_getstate(conn, action)

                result = controller.getController(state, geometry, conn)
                control = result[0]
                action = result[2]
                print("controller: ", control)
                print("T: ", result[1])
                print("action: ", action)

                for i in range(len(geometry["intersections"])):
                    network.applyControl(control[i],conn, geometry["intersections"][i])
                    print("Current traffic light is " + str(network.network[intersections[i]]["geometry"]["light_list"]))
           
        step += 1


   

    traci.close(False)

