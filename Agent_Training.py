from __future__ import absolute_import
from __future__ import print_function

import traci
import sys
import os
import keras
from Controller import dqnController
from Network import Network
from DQN_Agent import DQNAgent
from csv import writer


if __name__ == '__main__':
    

    # LOAD SUMO STUFF
    cfgfilename = "3_16RUNNER.sumo.cfg" # sys.argv[1]
    print("access config file " + str(cfgfilename))

    #os.environ['SUMO_HOME'] = "/usr/share/sumo"
    #if 'SUMO_HOME' in os.environ:
    #    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    #    sys.path.append(tools)
    #else:
    #    sys.exit("please declare environment variable 'SUMO_HOME'")

    dir_path = os.path.dirname(os.path.realpath(__file__))
    filepath = os.path.join(dir_path,"network",cfgfilename)
    filepath_step_data = os.path.join(dir_path,"step_Data.csv")
    filepath_data = os.path.join(dir_path,"Data.csv")
    print(filepath)
    #sumoCmd = ["sumo", "-c", filepath]
    sumoCmd = ["sumo-gui", "-c", filepath]  # if you want to see the simulation

    # parameters
    episodes = 2000
    batch_size = 32
    tracilabel = "sim1"

    # LOAD AGENT
    agent = DQNAgent()
    try:
        agent.load('DQN_control_3.h5')
        print('Agent_loaded')
    except:
        print('No models found')

    for e in range(episodes):
        
        step = 0
        Halting_number = 0
        total_avg_speed = 0
        last_ratio = 0
        next_ratio = 0
        total_reward = 0
        MSE = 0
        reward = 0
        action = 0

        # initialize the network object and controller object
        traci.start(sumoCmd, label=tracilabel)
        conn = traci.getConnection(tracilabel)

        network = Network(filepath, conn, agent)
        intersections = network.intersections
        training = dqnController()
        options = network.get_options()

        geometry = network.DQNgeometry

        print("Initial traffic light is " + str(geometry["DQN_light_list"]))

        while conn.simulation.getMinExpectedNumber() > 0 and step < 5000:
            conn.simulationStep()
            if step >= 0 and step%10 == 0:
                print('step: ' + str(step))
                if step > 2:
                    # memorizing      
                    total_reward += reward
                    new_state = network.DQN_getstate(conn, action)
                    agent.remember(state, action, reward, new_state, False)
                    print('memory_lenth: ' + str(len(agent.memory)))
                    if(len(agent.memory) > batch_size):
                        MSE = agent.replay(batch_size)
                    agent.target_train()

                    list01 = [step, reward, MSE]
                    with open(filepath_step_data, 'a', newline='') as w_object:
                        writer_object = writer(w_object)
                        writer_object.writerow(list01)
                        w_object.close()

                    reward = 0

                
                
                # controlling
                print("Current traffic light is " + str(network.network[intersections[0]]["geometry"]["light_list"]))
                 
                state = network.DQN_getstate(conn, action)
                
                result = training.getController(state, geometry, conn, agent)
                controller = result[0]
                action = result[1]
                print("controller: ", controller)
                print("action: ", action)

                network.applyControl(controller,conn, geometry["intersections"][0])
                print("Current traffic light is " + str(network.network[intersections[0]]["geometry"]["light_list"]))

            reward += -(network.getHaltingNum(conn))

            Halting_number += network.getHaltingNum(conn)

            speed = 0
            for i in range(len(network.allLaneId)):
                speed += conn.lane.getLastStepMeanSpeed(network.allLaneId[i])
            average_speed = speed/8
            total_avg_speed += average_speed

            
            step += 1


        mem = agent.memory[-1]
        del agent.memory[-1]
        agent.memory.append((mem[0], mem[1], reward, mem[3], True))
        #log.write('episode - ' + str(e) + ', total waiting time - ' +
        #          str(waiting_time) + ', static waiting time - 338798 \n')
        #log.close()
        print('episode - ' + str(e) + ' total waiting number - ' + str(Halting_number))

        #pyplot.show()

        agent.save('DQN_control_' + str(e) + '.h5')

        list = [str(e), Halting_number, total_reward, total_avg_speed]

        with open(filepath_data, 'a', newline='') as w_object:
            writer_object = writer(w_object)
            writer_object.writerow(list)
            print('list', list)
            w_object.close()

        print('finised')
        keras.backend.clear_session()

        
        traci.close(False)