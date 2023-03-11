'''
Author: sh0829kk 381534335@qq.com
Date: 2023-02-08 15:25:37
LastEditors: sh0829kk 381534335@qq.com
LastEditTime: 2023-02-15 15:59:15
FilePath: /AI-Traffic/test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# /usr/bin/python

class Data_Logger:

    def __init__(self, runID):
        #open csv
        self.lane_file = open("{}_metrics_lane.csv".format(runID), "w")
        self.veh_file = open("{}_metrics_veh.csv".format(runID), "w")

        self.lane_file.write('step,laneID,WaitingTime,CO2\n')
        self.veh_file.write('step,vehicleID,TimeLoss,AccumulatedWaitingTime\n')

    def updateLane(self, step, conn, allLaneId):

        for lane in allLaneId:
            self.lane_file.write("{},{},{},{}\n".format(
                step,
                lane,
                conn.lane.getWaitingTime(lane),
                conn.lane.getCO2Emission(lane)))
            # metrics[lane]['step'].append(step)
            # metrics[lane]['WaitingTime'].append(conn.lane.getWaitingTime(lane))
            # metrics[lane]['CO2'].append(conn.lane.getCO2Emission(lane))


    def updateVeh(self, step, conn, state):

        for vehicle in state["vehicleID"]:
            self.veh_file.write("{},{},{},{}\n".format(
                step,
                vehicle,
                conn.vehicle.getTimeLoss(vehicle),
                conn.vehicle.getAccumulatedWaitingTime(vehicle)))

            # metrics[vehicle]['step'].append(step)
            # metrics[vehicle]['TimeLoss'].append(
            #     conn.vehicle.getTimeLoss(vehicle))
            # metrics[vehicle]['AccumulatedWaitingTime'].append(conn.vehicle.getAccumulatedWaitingTime(vehicle))


    def close(self):
        self.lane_file.close()
        self.veh_file.close()


        # track time-step, vehicle-id, control-signal
    # metrics-land & metrics-vehicle
    #
