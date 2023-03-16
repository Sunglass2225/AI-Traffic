from __future__ import absolute_import
from __future__ import print_function

import traci
import sys
import os
import keras
from Controller import IDQNcontroller
from Network import Network
from DQN_Agent import DQNAgent
from csv import writer


if __name__ == '__main__':
    

    # LOAD SUMO STUFF
    cfgfilename = "SUMO_Network.sumo.cfg" # sys.argv[1]
    print("access config file " + str(cfgfilename))

    dir_path = os.path.dirname(os.path.realpath(__file__))
    filepath = os.path.join(dir_path,"network",cfgfilename)
    filepath_step_data = os.path.join(dir_path,"step_Data.csv")
    filepath_data = os.path.join(dir_path,"Data.csv")
    print(filepath)
    sumoCmd = ["sumo", "-c", filepath]
    #sumoCmd = ["sumo-gui", "-c", filepath]  # if you want to see the simulation in sumo-gui
        
    # parameters
    episodes = 2000
    batch_size = 32
    tracilabel = "sim1"

    # LOAD AGENT
    agent_list = [0, 0, 0]

    agent1 = DQNAgent()
    agent2 = DQNAgent()
    agent3 = DQNAgent()

    try:
        agent1.load('DQN_control_agent_1_49.h5')
        agent2.load('DQN_control_agent_2_49.h5')
        agent3.load('DQN_control_agent_3_49.h5')
        print('Agent_loaded')
    except:
        print('No models found')

    agent_list[0] = agent1
    agent_list[1] = agent2
    agent_list[2] = agent3

    for e in range(episodes):
        
        step = 0
        wating_time = 0
        Halting_number = 0
        total_reward = 0
        MSE = [0, 0, 0]
        reward_list = [0, 0, 0]
        state_list = [0, 0, 0]
        action_list = [0, 0, 0]
        state_check_list = [0, 0, 0]
        action_check_list = [0, 0, 0]

        # initialize the network object and controller object
        traci.start(sumoCmd, label=tracilabel)
        conn = traci.getConnection(tracilabel)

        network = Network(filepath, conn, agent_list[0])
        intersections = network.intersections
        controller = IDQNcontroller()

        while conn.simulation.getMinExpectedNumber() > 0 and step < 5000:
            conn.simulationStep()

            if step >= 0 and step%10 == 0:

                print('step: ' + str(step))                
                
                # memorizing
                if step > 2:
                    for i in range(len(intersections)):
                        total_reward += reward_list[i]
                        print('reward: ' + str(reward_list[i]))
                        new_state = network.IDQN_getstate(conn, intersection, action)
                        agent_list[i].remember(state_list[i], action_list[i], reward_list[i], new_state[0], False)
                        print('memory_lenth: ' + str(len(agent_list[i].memory)))
                        if(len(agent_list[i].memory) > batch_size):
                            MSE[i] = agent_list[i].replay(batch_size)
                        agent_list[i].target_train()


                # controlling
                for i in range(len(intersections)):         
                    intersection = intersections[i]
                    geometry = network.getGeometry(intersection)
                    action  = action_list[i]

                    #print(conn.multientryexit.getIDList())

                    state = network.IDQN_getstate(conn, intersection, action)
                    state_list[i] = state[0]

                    state_check_list[i] = state[1]

                    result = controller.getController(state[0], geometry, agent_list[i])
                    control = result[0]
                    action = result[1]
                    action_list[i] = action
                    print("   " + intersection + " light list : " + str(control))
                    print("   " + intersection + " Action : " + str(action))
                    # update the state of the network
                    network.applyControl(control,conn,intersection)   

                print("Current traffic light is " + str(network.network[intersections[0]]["geometry"]["light_list"]))
                print("Current traffic light is " + str(network.network[intersections[1]]["geometry"]["light_list"]))
                print("Current traffic light is " + str(network.network[intersections[2]]["geometry"]["light_list"]))

                list01 = [step, reward_list[0], reward_list[1], reward_list[2], action_list[0], action_list[1], action_list[2], MSE[0], MSE[1], MSE[2]]
                with open(filepath_step_data, 'a', newline='') as w_object:
                    writer_object = writer(w_object)
                    writer_object.writerow(list01)
                    w_object.close()

                for i in range(len(intersections)): 
                    reward_list[i] = 0
                

            
            for i in range(len(intersections)):    
                reward_list[i] += -(network.getIntersectionHaltingNum(intersections[i], conn))

            Halting_number += network.getHaltingNum(conn)            
            
            step += 1


        for i in range(len(intersections)): 
            agent_list[i].save('DQN_control_sinale_agent_' + str(i+1) + "_" + str(e) + '.h5')


        mem = agent_list[0].memory[-1]
        del agent_list[0].memory[-1]
        agent_list[0].memory.append((mem[0], mem[1], reward_list[-1], mem[3], True))
        
        print('episode - ' + str(e) + ' total Halting number - ' + str(Halting_number))

        list = [str(e), Halting_number, total_reward]

        with open(filepath_data, 'a', newline='') as w_object:
            writer_object = writer(w_object)
            writer_object.writerow(list)
            print('list', list)
            w_object.close()

        print('finised')
        keras.backend.clear_session()

        
        traci.close(False)