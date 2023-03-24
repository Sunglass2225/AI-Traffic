from abc import ABC, abstractmethod
from DQN_Agent import DQNAgent

 
class Controller(ABC):
    @abstractmethod
    def getController(self):
        pass

class MaxPressureController(Controller):
    def __init__(self):
        self.c = None

    # state :  
    def getController(self,geometry,state):
        for eachPair in geometry["pressure_map"]:                        # calculate pressure for each pair
            upper_lane = eachPair.split(',')[0]
            down_lane = eachPair.split(',')[1]
            numofvehicles_upper_lane = state["vehicle_number_each_lane"][upper_lane]
            numofvehicles_down_lane = state["vehicle_number_each_lane"][down_lane]
            length_upper_lane = geometry["length_lanes"][upper_lane]
            length_down_lane = geometry["length_lanes"][down_lane]
            geometry["pressure_map"][eachPair] = (numofvehicles_upper_lane / length_upper_lane) - (numofvehicles_down_lane / length_down_lane)
        print('geometry', geometry["pressure_map"])

        max_key = ""
        max_val = -1
        for key in geometry["pressure_map"]:
            if geometry["pressure_map"][key] > max_val:
                max_key = key
                max_val = geometry["pressure_map"][key]
    
        max_key_list = max_key.split(",")
        max_link = findItem(geometry["list_links"], max_key_list[0], max_key_list[1])  # defind the max prssure link
        max_phase = geometry["phase_matrix"][1][max_link[0]] # defind which phase the max prssure link belonge
        print('phase', geometry["phase_matrix"][1])
        print('max_phase', max_phase)
        
        coltroller = []
        for i in range(12):  # reset all light to RED
            coltroller.append('r')
        if max_phase == 0:
            coltroller[1] = 'G'
            coltroller[7] = 'G'
        if max_phase == 1:
            coltroller[4] = 'G'
            coltroller[10] = 'G'

        print(coltroller)
        return coltroller
        

class dqnController(Controller):
    def __init__(self):
        self.c = None

    def getController(self, state, geometry, conn, agent):
        action = agent.act(state)

        controller = []
        for x in range(len(geometry["DQN_light_list"][0])):  # reset all light to RED
            controller.append('r')
        
        if action == 0:
            controller[1] = 'G'
            controller[7] = 'G'
        if action == 1:
            controller[4] = 'G'
            controller[10] = 'G'
        
        return controller, action



def findItem(theList, item1, item2):
        return [(i) for (i, sub) in enumerate(theList) if item1 and item2 in sub]

