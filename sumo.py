"""
Traffic Technique - Adaptive Control Module

Authors: Georgios Terzoglou, Manos Nestwras  
Organization: Traffic Technique  
Date: 2024-12-29   

Description:  
This module interfaces with the SUMO traffic simulation tool.  
It manages SUMO configuration files, traffic light programs, offsets,  
and performs XML-based configurations for real-time traffic adaptation.  
"""


import os
import xml.etree.ElementTree as ET
import traci
import sys
import math
import logging
from datetime import datetime, timedelta

from custom_sumo import statisticxml

class SUMOHandler:
    """Class to manage SUMO-related XML manipulations and computations."""

    def __init__(self, config, env_config, peak_periods=None, threshold_stats=None):
        """
        Initialize the SUMOHandler class.
        Args:
            config (dict): Configuration settings for the SUMO process.
            env_config (dict): Environment configurations, including file paths.
        """
        self.config = config
        self.env_config = env_config
        self.sumo_path = env_config["paths"]["sumoPATH"]
        self.peak_periods = peak_periods
        self.threshold_stats = threshold_stats

    def write_paths_to_sumo_cfg(self, sumocfg, netfile, routefiles, additionalfiles, statisticoutput, guisettingsfile):
        """
        Update paths in the SUMO configuration file.
        """
        filename = os.path.join(self.sumo_path, sumocfg)
        xml_tree = ET.parse(filename)
        root_element = xml_tree.getroot()

        for element in root_element:
            for child in element:
                match child.tag:
                    case 'net-file':
                        child.attrib['value'] = netfile
                    case 'route-files':
                        child.attrib['value'] = routefiles
                    case 'additional-files':
                        child.attrib['value'] = additionalfiles
                    case 'statistic-output':
                        child.attrib['value'] = statisticoutput
                    case 'gui-settings-file':
                        child.attrib['value'] = guisettingsfile

        xml_tree.write(filename, encoding='UTF-8', xml_declaration=True)

    def tlstoxml(self, id, programID, pos, duration):
        """
        Update the duration of traffic light phases in `tlsProgram.xml`.

        Args:
            id (str): Traffic light ID.
            programID (list): List of program IDs.
            pos (int): Position of the traffic light phase.
            duration (int): Duration to set for the traffic light phase.
        """
        filename = os.path.join(self.sumo_path, "tlsProgram.xml")
        xml_tree = ET.parse(filename)
        root_element = xml_tree.getroot()

        for PRid in programID:
            position = -1
            for index in root_element:
                for element in index:
                    if index.attrib['id'] == str(id) and index.attrib['programID'] == str(programID[PRid]):
                        position += 1
                        if pos == position:
                            element.attrib['duration'] = str(duration)
                            # Uncomment the following line if you need to change the state:
                            # element.attrib['state'] = str(phase)
                            if duration < 0:
                                print(f"You have negative values at position {position} with id {id} and programID {programID[PRid]}")
                            elif duration == 0:
                                element.attrib['duration'] = '0'
                                print(f"You have zero values at position {position} with id {id} and programID {programID[PRid]}")

            xml_tree.write(filename)

        
    def tlstoxml_offset(self, id, program_ids, offset):
        """
        Update the offset value for the traffic light program in `tlsProgram.xml`.
        """
        filename = os.path.join(self.sumo_path, "tlsProgram.xml")
        xml_tree = ET.parse(filename)
        root_element = xml_tree.getroot()

        for program_id in program_ids:
            for index in root_element:
                if index.attrib['id'] == str(id) and index.attrib['programID'] == str(program_id):
                    index.attrib['offset'] = str(offset)

            xml_tree.write(filename)

    def offsetfromxml(self, id, program_id):
        """
        Retrieve the offset value from the traffic light program in `tlsProgram.xml`.
        """
        filename = os.path.join(self.sumo_path, "tlsProgram.xml")
        xml_tree = ET.parse(filename)
        root_element = xml_tree.getroot()

        for index in root_element:
            if index.attrib['id'] == id and index.attrib['programID'] == program_id:
                return int(index.attrib['offset'])

    def flowtoxml(self, flows):
        """
        Update flows in `dataEdge.xml`.
        """
        filename = os.path.join(self.sumo_path, "dataEdge.xml")
        xml_tree = ET.parse(filename)
        root_element = xml_tree.getroot()

        for element in root_element:
            for flow in flows:
                if element.attrib['id'] == flow[0]:
                    element.attrib["entered"] = str(flow[1])

        xml_tree.write(filename)

    def turnstoxml(self, flows):
        filename = self.sumo_path + "turns.xml"
        xmlTree = ET.parse(filename)
        rootElement = xmlTree.getroot()
        indexes= rootElement
        for index in indexes:
            for element in index: 
                for i in range(len(flows)):
                    if element.attrib['from'] == flows[i][0] and element.attrib['to'] == flows[i][1]:
                        element.attrib["count"] = str(flows[i][2])

        xmlTree.write(filename)

    def flowtoxml(self, flows):
        filename = self.sumo_path + "dataEdge.xml"
        xmlTree = ET.parse(filename)
        rootElement = xmlTree.getroot()
        indexes= rootElement
        for index in indexes:
            for element in index: 
                for i in range(len(flows)):
                    if element.attrib['id'] == flows[i][0]:
                        element.attrib["entered"] = str(flows[i][1])

        xmlTree.write(filename)

    def sumoPark(self, tcID, park_area, activate):
        """
        Manage parking area settings in `parkingArea.xml`.
        """
        filename = os.path.join(self.sumo_path, "parkingArea.xml")
        xml_tree = ET.parse(filename)
        root_element = xml_tree.getroot()

        stop_edge = self.config[str(tcID)]["sumoPark"][park_area]["edge_stop"]
        go_edge = self.config[str(tcID)]["sumoPark"][park_area]["edge_go"]

        for index in root_element:
            for element in index:
                if element.tag == 'route' and (element.attrib['edges'] == stop_edge or element.attrib['edges'] == go_edge):
                    element.attrib['edges'] = go_edge if activate == 0 else stop_edge
                if element.tag in ['go', 'stop'] and element.attrib['parkingArea'] == park_area:
                    element.tag = 'go' if activate == 0 else 'stop'

        xml_tree.write(filename, encoding='UTF-8', xml_declaration=True)

    def closedLane(self, tcID, ID, data):
        """
        Manage closed lanes based on turn flows.
        """
        if not self.config[str(ID)].get("sumoPark"):
            return

        for i in range(len(data)):
            for j in range(len(data[i])):
                globals()[f"tcID{tcID[i]}H{j + 1}"] = data[i][j]

        for park_area in self.config[str(ID)]["sumoPark"]:
            activate = round(eval(self.config[str(ID)]["sumoPark"][park_area]["activate"]))
            self.sumoPark(ID, park_area, activate)

        for i in range(len(data)):
            for j in range(len(data[i])):
                del globals()[f"tcID{tcID[i]}H{j + 1}"]

    @staticmethod
    def extract_sensor_name(sensor_expr):
        # Extract H# from something like "tcID3H1"
        import re
        match = re.search(r"(H\d+)", sensor_expr)
        return match.group(1) if match else sensor_expr

    #@staticmethod
    # def peak_eval(sensor_name, value, timestamp, peak_df, sensor_thresholds):
    #     """
    #     Return the peak-adjusted value based on whether the current timestamp
    #     is within the peak period for the given sensor.
    #     """
    #     day = timestamp.strftime('%A')
    #     time_str = timestamp.strftime('%H:%M')
    #     # Check if this sensor is in a peak period
    #     in_peak = False
    #     if peak_df is not None:
    #         matches = peak_df[(peak_df['Sensor'] == sensor_name) & (peak_df['Day'] == day)]
    #         for _, row in matches.iterrows():
    #             if row['Start Time'] <= time_str <= row['End Time']:
    #                 in_peak = True
    #                 break

    #     # Return value based on peak condition
    #     if in_peak:
    #         threshold = sensor_thresholds.get(sensor_name, {}).get('threshold', value)
    #         return max(threshold, value)
    #     else:
    #         return value

    @staticmethod 
    def peak_eval(sensor_name, value, timestamp, peak_df, sensor_thresholds):
        """
        Return the peak-adjusted value based on whether the current timestamp
        is within the peak period for the given sensor.
        Logs to 'peak_eval_log.log' only when threshold > value.
        """
        
        # Setup dedicated logger for peak_eval
        logger = logging.getLogger("PeakLogger")
        if not logger.handlers:
            handler = logging.FileHandler('peak_eval_log.log', mode='w')  # Overwrite only on first call
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            logger.propagate = False  # prevent logging from being passed to the root logger
    
        day = timestamp.strftime('%A')
        time_str = timestamp.strftime('%H:%M')
    
        # Check if this sensor is in a peak period
        in_peak = False
        if peak_df is not None:
            matches = peak_df[(peak_df['Sensor'] == sensor_name) & (peak_df['Day'] == day)]
            for _, row in matches.iterrows():
                if row['Start Time'] <= time_str <= row['End Time']:
                    in_peak = True
                    break

        # Return value based on peak condition
        if in_peak:
            threshold = sensor_thresholds.get(sensor_name, {}).get('threshold', value)
    
            if threshold > value:
                logger.info(
                    f"Sensor: {sensor_name} | Threshold: {threshold} > Value: {value} | Timestamp: {timestamp}"
                )

            return max(threshold * 0.8, value)   # weight factor
        else:
            return value


    def sumoFlows(self, tcID, ID, data, peak_periods_dataframe, threshold_stats_dataframe):
        """
        Calculate and return flows for the SUMO simulation.
        """
        for i in range(len(data)):
            for j in range(len(data[i])):
                globals()[f"tcID{tcID[i]}H{j + 1}"] = data[i][j]

        timestamp = datetime.now()

        eval_globals = globals().copy() 
        eval_globals.update({
            "timestamp": timestamp,
            "peak": lambda sensor_expr: SUMOHandler.peak_eval(
                sensor_name=self.extract_sensor_name(sensor_expr),
                value=eval(sensor_expr, eval_globals),
                timestamp=timestamp,
                peak_df=peak_periods_dataframe,
                sensor_thresholds=threshold_stats_dataframe
            )
        })

        flows = [
            [flow_name, max(0, round(eval(self.config[str(ID)]["sumoFlows"][flow_name], eval_globals)))]
            for flow_name in self.config[str(ID)]["sumoFlows"]
        ]

        for i in range(len(data)):
            for j in range(len(data[i])):
                del globals()[f"tcID{tcID[i]}H{j + 1}"]

        return flows

    def sumoTurns(self, tcID, ID, data, peak_periods_dataframe, threshold_stats_dataframe):
        """
        Calculate and return turn counts for the SUMO simulation.
        """
        if not self.config[str(ID)].get("sumoTurns"):
            return []

        for i in range(len(data)):
            for j in range(len(data[i])):
                globals()[f"tcID{tcID[i]}H{j + 1}"] = data[i][j]

        timestamp = datetime.now()

        eval_globals = globals().copy() 
        eval_globals.update({
            "timestamp": timestamp,
            "peak": lambda sensor_expr: SUMOHandler.peak_eval(
                sensor_name=self.extract_sensor_name(sensor_expr),
                value=eval(sensor_expr, eval_globals),
                timestamp=timestamp,
                peak_df=peak_periods_dataframe,
                sensor_thresholds=threshold_stats_dataframe
            )
        })

        turns = [
            [
                turn["from"],
                turn["to"],
                max(0, round(eval(turn["count"], eval_globals)))
            ]
            for turn in self.config[str(ID)]["sumoTurns"].values()
        ]

        for i in range(len(data)):
            for j in range(len(data[i])):
                del globals()[f"tcID{tcID[i]}H{j + 1}"]

        return turns

    def stageFlows(self, tcID, ID, data, peak_periods_dataframe, threshold_stats_dataframe):
        """
        Calculate and return traffic stage flows.
        """
        data_hourly = [[value * 12 for value in detector] for detector in data]

        # Dynamically create variables using globals()
        for i in range(len(data_hourly)):
            for j in range(len(data_hourly[i])):
                globals()[f"tcID{tcID[i]}H{j + 1}"] = data_hourly[i][j]

        timestamp = datetime.now()
        eval_globals = globals().copy() 
        eval_globals.update({
            "timestamp": timestamp,
            "peak": lambda sensor_expr: SUMOHandler.peak_eval(
                sensor_name=self.extract_sensor_name(sensor_expr),
                value=eval(sensor_expr, eval_globals),
                timestamp=timestamp,
                peak_df=peak_periods_dataframe,
                sensor_thresholds=threshold_stats_dataframe
            )
        })
    
        stages = [
            max(0, round(eval(self.config[str(ID)]["tr_stages"][stage_name], eval_globals)))
            for stage_name in self.config[str(ID)]["tr_stages"]
        ]
        
        total_traffic = sum(stages)

        # Cleanup globals() 
        for i in range(len(data_hourly)):
            for j in range(len(data_hourly[i])):
                del globals()[f"tcID{tcID[i]}H{j + 1}"]

        return stages, total_traffic

    def compute_sumo_and_algo_flows(self, tcID, data_arr, sumo_data, tr_stg, global_traffic):
        """
        Compute all SUMO and algorithmic flows, updating the provided data structures.
        """
        for i in range(len(tcID)):
            sumo_data["sumo_flows"].extend(self.sumoFlows(tcID, tcID[i], data_arr, self.peak_periods[i], self.threshold_stats[i]))
            sumo_data["sumo_turns"].extend(self.sumoTurns(tcID, tcID[i], data_arr, self.peak_periods[i], self.threshold_stats[i]))
            self.closedLane(tcID, tcID[i], data_arr)
            stage, traffic = self.stageFlows(tcID, tcID[i], data_arr, self.peak_periods[i], self.threshold_stats[i])
            tr_stg.append(stage)
            global_traffic.append(traffic)
        
        return tr_stg, global_traffic, sumo_data
    
    def run(self):

        # main loop. do something every simulation step until no more vehicles are
        # loaded or running
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
                    
        sys.stdout.flush()
        traci.close()

        return statisticxml()

    
    def des_steps(self, tcID, cycle_length, tr_stg, global_traffic, lstgr):
        """
        Calculate decision steps for traffic lights.

        Args:
            tcID (str): Traffic control ID.
            cycle_length (int): Length of the traffic cycle.
            tr_stg (list): Traffic stages data.
            global_traffic (int): Global traffic volume.
            lstgr (int): Lost green time.

        Returns:
            list: Decision steps for the traffic control stages.
        """
        agt = cycle_length - self.config[str(tcID)]["dead_time"]
        green_to_split = agt - lstgr

        dsteps = [0] * len(self.config[str(tcID)]["stages"])
        i = 0
        for stage_name in self.config[str(tcID)]["stages"]:
            stage = self.config[str(tcID)]["stages"][stage_name]
            dsteps[i] = self.split(
                green_to_split, global_traffic, tr_stg[i],
                stage["trs_sec"], stage["fixed_green"], stage["minTime"], stage["lost_green"]
            )
            agt -= stage["fixed_green"]
            i += 1

        # Repeat distribution until all minimum step values are met
        for _ in range(len(self.config[str(tcID)]["stages"]) + 1):
            dsteps = self.distrib(agt, dsteps)

            i = 0
            for stage_name in self.config[str(tcID)]["stages"]:
                min_time = self.config[str(tcID)]["stages"][stage_name]["minTime"]
                if dsteps[i] < min_time:
                    dsteps[i] = -min_time
                i += 1

            if all(d > 0 for d in dsteps):
                return dsteps

            i = 0
            for stage_name in self.config[str(tcID)]["stages"]:
                min_time = self.config[str(tcID)]["stages"][stage_name]["minTime"]
                if dsteps[i] <= min_time:
                    dsteps[i] = -min_time
                i += 1

        # Final fallback
        i = 0
        for stage_name in self.config[str(tcID)]["stages"]:
            min_time = self.config[str(tcID)]["stages"][stage_name]["minTime"]
            if dsteps[i] < min_time:
                dsteps[i] = min_time
            i += 1

        return dsteps
    
    def calculate_decision_steps(self, tcID, cycle_length, tr_stg, global_traffic, lstgr):
        """Calculate decision steps and load them into XML."""
        steps_arr = []
        for i in range(len(tcID)):
            steps = self.des_steps(tcID[i], cycle_length, tr_stg[i], global_traffic[i], lstgr[i])
            steps_arr.append(steps)
            for k, stage_name in enumerate(self.config[str(tcID[i])]["stages"]):
                
                self.tlstoxml(
                    tcID[i], self.config[str(tcID[i])]["sumoPrIDs"],
                    self.config[str(tcID[i])]["stages"][stage_name]["des_step"], steps[k]
                )
        return steps_arr

    def offsets(self, tcID, ID, cycle_length, decision_steps):
        """Calculate and return offsets."""
        for i in range(len(decision_steps)):
            for j in range(len(decision_steps[i])):
                globals()[f"tcID{tcID[i]}dStep{j+1}"] = decision_steps[i][j]

        adaptOffset = eval(self.config[str(ID)][str(cycle_length)]["offset"])
        if adaptOffset < 0:
            adaptOffset += cycle_length
        elif adaptOffset > cycle_length:
            adaptOffset -= cycle_length

        sumoOffset = eval(self.config[str(ID)][str(cycle_length)]["sumoOffset"])
        if sumoOffset < 0:
            sumoOffset += cycle_length
        elif sumoOffset > cycle_length:
            sumoOffset -= cycle_length

        # Clean up global variables
        for i in range(len(decision_steps)):
            for j in range(len(decision_steps[i])):
                del globals()[f"tcID{tcID[i]}dStep{j+1}"]

        return adaptOffset, sumoOffset

    def calculate_offsets(self, tcID, cycle_length, decision_steps):
        """Calculate offsets and load them into XML."""
        offsets_c_arr = []
        offsetsSumo_c_arr = []
        for i in range(len(tcID)):
            offset, offsetSumo = self.offsets(tcID, tcID[i], cycle_length, decision_steps)
            offsets_c_arr.append(offset)
            offsetsSumo_c_arr.append(offsetSumo)
            self.tlstoxml_offset(tcID[i], self.config[str(tcID[i])]["sumoPrIDs"], offsetSumo)
        return offsets_c_arr, offsetsSumo_c_arr
    
    def split(self, green_time, global_traffic, tr_stg, trs_sec, fixed_green, minTime, lost_green):
        """
        Calculate the split (decision step) for a given traffic stage.

        Args:
            green_time (int): Total green time to be distributed.
            global_traffic (int): Total traffic volume.
            tr_stg (int): Traffic volume for the current stage.
            trs_sec (int): Transition seconds for the stage.
            fixed_green (int): Fixed green time for the stage.
            minTime (int): Minimum green time for the stage.
            lost_green (int): Lost green time for the stage.

        Returns:
            int: Decision step for the stage.
        """
        if global_traffic > 0:
            stage_traffic_proportion = tr_stg / global_traffic
            available_green_time = (trs_sec / 2) * green_time
            ideal_green_split = round(stage_traffic_proportion * available_green_time)
            decision_step = ideal_green_split - fixed_green + lost_green
        else:
            decision_step = 1

        if decision_step <= minTime:
            decision_step = -minTime

        return decision_step

    def distrib(self, agtn, des_steps):
        """
        Distribute available green time proportionally among stages.

        Args:
            agtn (int): Available green time to distribute.
            des_steps (list): Current decision steps.

        Returns:
            list: Adjusted decision steps after distribution.
        """
        posgr = 0
        prop = []

        for i in range(len(des_steps)):
            if des_steps[i] < 0:
                agtn += des_steps[i]
            else:
                posgr += des_steps[i]

        for i in range(len(des_steps)):
            if des_steps[i] > 0:
                prop.append(des_steps[i] / posgr)

        distributed_steps = self.distribute_elements_in_slots(agtn, prop)

        j = 0
        for i in range(len(des_steps)):
            if des_steps[i] < 0:
                des_steps[i] = -des_steps[i]
            else:
                des_steps[i] = distributed_steps[j]
                j += 1

        for i in range(len(des_steps)):
            if des_steps[i] < 1:
                des_steps[i] = 1

        return des_steps

    def distribute_elements_in_slots(self, total, prop):
        """
        Distribute a total number proportionally among slots.

        Args:
            total (int): Total number to distribute.
            prop (list): Proportions for each slot.

        Returns:
            list: Distributed elements.
        """
        solid = [int(total * p) for p in prop]
        short = [total * p - solid[i] for i, p in enumerate(prop)]

        leftover = int(round(sum(short)))

        for _ in range(leftover):
            if min(solid) < 1:
                shortest_slot = solid.index(min(solid))
            else:
                shortest_slot = short.index(max(short))

            solid[shortest_slot] += 1
            short[shortest_slot] = 0

        return solid
    
def get_paths_for_predictions(timestamp, project, tcID):
    """
    Generate file paths for predictions based on a timestamp, project, and task/component ID.

    Args:
        timestamp (datetime): The current time for generating paths.
        project (str): The project name to include in the paths.
        tcID (int): The unique identifier for the task/component.

    Returns:
        tuple: Paths for the previous 5-minute interval and the current timestamp.
    """
    def format_path(base_time):
        return (
            f"logs/{project}/predictions/"
            f"{base_time.year}/{base_time.month:02d}/{base_time.day:02d}/"
            f"{base_time.strftime('%H-%M')}/tcID{tcID}"
        )

    # Adjust to the nearest 5-minute interval
    previous_time = timestamp - timedelta(minutes=5)

    # Generate paths
    path_prev = format_path(previous_time)
    path_cur = format_path(timestamp)

    return path_prev, path_cur