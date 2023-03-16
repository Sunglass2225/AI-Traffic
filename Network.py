import pandas as pd
import numpy as np
from sumolib import checkBinary
import traci
import os, sys
import optparse

class Network:

  def __init__(self,cfgfilename, conn, agent):
    
    self.network = {}
    self.DQNgeometry = {}
    self.allLaneId = []
    step = 0
    i = 0
    allLaneId = []
    intersections = conn.trafficlight.getIDList()
    for i in range(len(intersections)):
      self.network[intersections[i]] = {"geometry": {}, "state" : {}}

      list_links = trafficlight_link(intersections[i], conn)

      LaneID = getLaneID(list_links)
      for k in range(len(LaneID)):
        allLaneId.append(LaneID[k])
      numberOfLane = len(LaneID)
      allnumberofLane = len(set(allLaneId))
      conn.trafficlight.setRedYellowGreenState(intersections[i], "GGrrrrrrrrrr")

      light_list = trafficlight_light(intersections[i], conn)

      phase_matrix = trafficlight_phase(list_links, light_list, agent)

      length_lanes = {}   # map from lane_id to length of it
      lane_pairs = {}     # map from upper_lane_id to down_lane_id
      pressure_map = {}   # map from pair of lanes'( one stream)' to pressure
      vehicles_lanes = []
      res = []


      for j in range(len(LaneID)):
        each_length = conn.lane.getLength(LaneID[j]) # get current length
        length_lanes[LaneID[j]] = each_length # put into a map
                  
      # forming the map from pair of lanes'( one stream)' to the number of vehicles         
      for x in range(len(list_links)):                                      
          upper_to_down = list_links[x][0] + "," + list_links[x][1]   # forming the key: upper_lane_id + comma + down_lane_id
          pressure_map[upper_to_down] = 0
      step += 1


      self.network[intersections[i]]["geometry"]["LaneID"] = LaneID
      self.network[intersections[i]]["geometry"]["pressure_map"] = pressure_map
      self.network[intersections[i]]["geometry"]["length_lanes"] = length_lanes
      self.network[intersections[i]]["geometry"]["list_links"] = list_links
      self.network[intersections[i]]["geometry"]["phase_matrix"] = phase_matrix
      self.network[intersections[i]]["geometry"]["numberOfLane"] = numberOfLane
      self.network[intersections[i]]["geometry"]["length_lanes"] = length_lanes
      self.network[intersections[i]]["geometry"]["light_list"] = light_list

    DQN_list_links = []
    DQN_light_list = []
    DQN_phase_matrix = {}
    for i in range(len(intersections)):
        DQN_list_links.append(trafficlight_link(intersections[i],conn))
        DQN_light_list.append(trafficlight_light(intersections[i],conn))
        DQN_phase_matrix[intersections[i]] = trafficlight_phase(DQN_list_links[i], DQN_light_list[i], agent)

    self.allLaneId = list(set(allLaneId))
    self.allnumberofLane = allnumberofLane
    self.intersections = intersections
    self.DQNgeometry["DQN_list_links"] = DQN_list_links
    self.DQNgeometry["DQN_light_list"] = DQN_light_list
    self.DQNgeometry["DQN_phase_matrix"] = DQN_phase_matrix
    self.DQNgeometry["intersections"] = intersections
     
  def getGeometry(self,intersection):
    return self.network[intersection]["geometry"]

  def getState(self,conn,intersection):
    vehicle_number_each_lane = {}                # map from lane_id to number of vehicles
    for x in range(self.network[intersection]["geometry"]["numberOfLane"]):  
        lane_length = conn.lane.getLength(self.network[intersection]["geometry"]["LaneID"][x])                    # extract length and number of
        total_number = conn.lane.getLastStepVehicleNumber(self.network[intersection]["geometry"]["LaneID"][x])    # vehicles in each lane
        vehicle_number_each_lane[self.network[intersection]["geometry"]["LaneID"][x]] = total_number
    self.network[intersection]["state"]["vehicle_number_each_lane"] = vehicle_number_each_lane
    VehicleID = conn.vehicle.getIDList()
    self.network[intersection]["state"]["vehicleID"] = VehicleID
    return self.network[intersection]["state"]

  def IDQN_getstate(self, conn, intersection, action):
    number_each_lane = {}
    for x in range(len(self.network[intersection]["geometry"]["LaneID"])):  
        total_number = conn.lane.getLastStepVehicleNumber(self.network[intersection]["geometry"]["LaneID"][x])
        number_each_lane[self.network[intersection]["geometry"]["LaneID"][x]] = total_number
            
    number_each_lane_list = list(number_each_lane.values())            

    DQN_action =[]
    for x in range(4):   
        if x == action :
            DQN_action.append(1)
        else:
            DQN_action.append(0)


    num_each_lane_arr = np.array(number_each_lane_list)
    num_each_lane_arr = num_each_lane_arr.reshape(1, 24, 1)

    DQN_action_arr = np.array(DQN_action)
    DQN_action_arr = DQN_action_arr.reshape(1, 4, 1)
      
    return [num_each_lane_arr, DQN_action_arr], number_each_lane_list

  def getVehicleNum(self, conn):
    VehicleNum = 0
    for i in range(self.allnumberofLane):
        VehicleNum += conn.lane.getLastStepVehicleNumber(self.allLaneId[i])

    return VehicleNum

  def getHaltingNum(self, conn):
    HaltingNum = 0
    for i in range(self.allnumberofLane):
        HaltingNum += conn.lane.getLastStepHaltingNumber(self.allLaneId[i])

    return HaltingNum

  def getIntersectionHaltingNum(self, intersection, conn):
    HaltingNum = 0
    for i in range(len(self.network[intersection]["geometry"]["LaneID"])):
        HaltingNum += conn.lane.getLastStepHaltingNumber(self.network[intersection]["geometry"]["LaneID"][i])

    return HaltingNum

  def gethaltingratio(self, intersection, conn):
    pressuer_ratio = 0
    veh_num = 0
    hal_num = 0
    for x in range(len(self.network[intersection]["geometry"]["LaneID"])):
        veh_num += conn.lane.getLastStepVehicleNumber(self.network[intersection]["geometry"]["LaneID"][x])
        hal_num += conn.lane.getLastStepHaltingNumber(self.network[intersection]["geometry"]["LaneID"][x])
    
    if veh_num == 0:
        pressuer_ratio = 0
    else:
        pressuer_ratio = hal_num/veh_num

    return pressuer_ratio
  
  def applyControl(self,controller,conn,intersection):
    RedYellowGreenState = ''.join(str(e) for e in controller)
    conn.trafficlight.setRedYellowGreenState(intersection, RedYellowGreenState)
    self.network[intersection]["geometry"]["light_list"] = controller
     
  
     


#########################################################################################
##    helper method for extracting information of the network
def getLaneNumber(idList):
  res = 0
  for i in range(len(idList)):
      if idList[i][0] != ':':
          res = res + 1
  return res

def findItem(theList, item1, item2):
  return [(i) for (i, sub) in enumerate(theList) if item1 and item2 in sub]

def getLaneID(links):
  res = []
  for i in range(len(links)):
    res.append(links[i][0])
    res.append(links[i][1])
  return res

def trafficlight_link(junction,conn):
  links = conn.trafficlight.getControlledLinks(junction)
  out = [item for t in links for item in t]
  list_links = [list(ele) for ele in out]
  for i in range(len(list_links)):
      list_links[i].pop(2)
      # ex:[['5_0', '-6_0'], ['5_1', '-6_1'], ['5_2', '2_2'], ['-2_0', '-1_0'], ['-2_1', '-1_1'], ['-2_2', '-6_2'], ['6_0', '-5_0'], ['6_1', '-5_1'], ['6_2', '-1_2'], ['1_0', '2_0'], ['1_1', '2_1'], ['1_2', '-5_2']]

  return list_links

def trafficlight_light(junction,conn):
  lights = conn.trafficlight.getRedYellowGreenState(junction)
  light_list = list(lights)
  # ex:['r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r']
  
  return light_list

def trafficlight_phase(list_links, light_list, agent):  #putting the link, phase, and light into a matrix
  a = 0  # phase id
  w = len(list_links)
  h = 3
  Matrix = [[0 for x in range(w)] for y in range(h)] 
  #print(Matrix)
  
  for i in range(len(list_links)):
      Matrix[0][i] = list_links[i]
      
  for i in range(len(list_links)):
      if list_links[i][0][0] == list_links[i-1][0][0] and list_links[i][0][1] == list_links[i-1][0][1] and list_links[i][1][0] == list_links[i-1][1][0] and list_links[i][1][1] == list_links[i-1][1][1]:
          a -= 1
          # using the fisrt two chart of the two elements to defined whether their in the same phase
      
      Matrix[1][i] = a % agent.action_size
      a += 1

      
  for i in range(len(list_links)):
      Matrix[2][i] = light_list[i]
      
  return Matrix
