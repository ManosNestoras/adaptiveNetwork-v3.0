import xml.etree.ElementTree as ET
from dotenv import load_dotenv
import traci
import sys
import os
from routeSampler import runRouteSampler
from routeSampler import get_options as RS_get_options

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

# load enviroment parameters
curr_path = current_directory = os.getcwd()
load_dotenv(curr_path + '/.env')
filePATH = os.getenv('filePATH')
PATH = os.getenv('sumoPATH')
sumocfg = os.getenv('sumocfg')

# read offset from tlsProgram.xml
def offsetfromxml(id,programID):
    filename = PATH + "tlsProgram.xml"
    xmlTree = ET.parse(filename)
    rootElement = xmlTree.getroot()
    indexes = rootElement
    for index in indexes:
        if index.attrib['id'] == id and index.attrib['programID'] == programID:
            offset = index.attrib['offset']

    return int(offset)

# read the SUMO results from statistics.xml
def statisticxml():
    file= PATH + "statistics.xml"
    tree = ET.parse(file)
    rootElement = tree.getroot()
    indexes =rootElement
    for index in indexes:
                if index.tag == "vehicleTripStatistics":
                    timeloss = (index.attrib['timeLoss'])
                    j=0
    for index in indexes:
                if index.tag == "vehicleTripStatistics":
                    speed = (index.attrib['speed'])
                    j=0
    for index in indexes:
                if index.tag == "vehicleTripStatistics":
                    duration = (index.attrib['duration'])
                    j=0
    
    return([timeloss, speed, duration])

'''# global variables for TraCI
# the id of controlled intersections
TLSID1 = '1'
TLSID3 = '3'

# pedestrian edges at the controlled intersections
WALKINGAREAS_ID1_21 = [":1_w0",":1_w1"]
CROSSINGS_ID1_21 = [":1_c0"]'''
      
def run():   
    """execute the TraCI control loop"""
    #********************** initialization *********************************
    '''# flags
    flg_ID1_2 = 0
    flg_ID1_4 = 0
   
    # calls
    activeRequest_ID1_21 = False
    ID1_D1 = 0
    ID1_R1 = 0

    # start program 
    traci.trafficlight.setProgram(TLSID1, "1")
    traci.trafficlight.setPhase(TLSID1, 0)
    # set offset
    offset = offsetfromxml(TLSID1, "1")
    stp_dur = traci.trafficlight.getPhaseDuration(TLSID1)
    traci.trafficlight.setPhaseDuration(TLSID1, stp_dur + offset)

    # start program
    traci.trafficlight.setProgram(TLSID3, "1")
    traci.trafficlight.setPhase(TLSID3, 0)
    # set offset
    offset = offsetfromxml(TLSID3, "1")
    stp_dur = traci.trafficlight.getPhaseDuration(TLSID3)
    traci.trafficlight.setPhaseDuration(TLSID3, stp_dur + offset)'''

    # main loop. do something every simulation step until no more vehicles are
    # loaded or running
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
    
    '''#**********************ID1 script**************************************
        # check wether there is a waiting pedestrian 
        if not activeRequest_ID1_21:
            activeRequest_ID1_21 = checkWaitingPersons_ID1_21()
            #print("activeRequest_ID1_21", activeRequest_ID1_21)

        #phase 2
        if traci.trafficlight.getPhase(TLSID1) == 2 and flg_ID1_4 == 1:
            flg_ID1_4 = 0
            if traci.trafficlight.getProgram(TLSID1) == "22":
                # switch program
                traci.trafficlight.setProgram(TLSID1, "1")
                traci.trafficlight.setPhase(TLSID1, 2)
                #print("ID1 jump to Pr1") 

        #phase 4
        if traci.trafficlight.getPhase(TLSID1) == 4 and flg_ID1_2 == 0:
            flg_ID1_2 = 1
            # check detector if vehicles are waiting
            ID1_D1 = traci.inductionloop.getLastStepVehicleNumber("ID1_D1_0")
            if traci.trafficlight.getProgram(TLSID1) == "1":
                if ID1_D1 > 0 and activeRequest_ID1_21 == False:
                    # switch program
                    traci.trafficlight.setProgram(TLSID1, "21")
                    traci.trafficlight.setPhase(TLSID1, 4)
                    #print("ID1 jump to Pr21") 
                if ID1_D1 == 0 and activeRequest_ID1_21 == False:
                    # switch program
                    traci.trafficlight.setProgram(TLSID1, "22")
                    traci.trafficlight.setPhase(TLSID1, 4)
                    #print("ID1 jump to Pr22") 

        #phase 12
        if traci.trafficlight.getPhase(TLSID1) == 12:
            if traci.trafficlight.getProgram(TLSID1) == "1":
                # reset state   
                activeRequest_ID1_21 = False
                #print("reset activeRequest_ID1_21")

        #phase 15
        if traci.trafficlight.getPhase(TLSID1) == 15 and flg_ID1_2 == 1:
            flg_ID1_2 = 0
            if traci.trafficlight.getProgram(TLSID1) == "21":
                # switch program
                traci.trafficlight.setProgram(TLSID1, "1")
                traci.trafficlight.setPhase(TLSID1, 15)
                #print("ID1 jump to Pr1") 
            if traci.trafficlight.getProgram(TLSID1) == "22":
                # switch program
                traci.trafficlight.setProgram(TLSID1, "1")
                traci.trafficlight.setPhase(TLSID1, 15)
                #print("ID1 jump to Pr1") 

        #phase 19
        if traci.trafficlight.getPhase(TLSID1) == 19 and flg_ID1_4 == 0:
            flg_ID1_4 = 1
            # check detector if vehicles are waiting
            ID1_R1 = traci.inductionloop.getLastStepVehicleNumber("ID1_R1_0")
            if traci.trafficlight.getProgram(TLSID1) == "1":
                if ID1_R1 == 0:
                    # switch program
                    traci.trafficlight.setProgram(TLSID1, "22")
                    traci.trafficlight.setPhase(TLSID1, 19)
                    #print("ID1 jump to Pr22") 
    #**********************ID1 script end**********************************

    #**********************ID3 script*************************************
    #**********************ID3 script end**********************************'''
                
    sys.stdout.flush()
    traci.close()

    return statisticxml()

'''def checkWaitingPersons_ID1_21():
    """check whether a person has requested to cross the street"""
    # check both sides of the crossing
    for edge in WALKINGAREAS_ID1_21:
        peds = traci.edge.getLastStepPersonIDs(edge)
        # check who is waiting at the crossing
        # we assume that pedestrians push the button upon
        # standing still for 1s
        for ped in peds:
            if (traci.person.getWaitingTime(ped) == 1 and
                    traci.person.getNextEdge(ped) in CROSSINGS_ID1_21):
                #numWaiting_21 = traci.trafficlight.getServedPersonCount(TLSID5, PEDESTRIAN_GREEN_PHASE_ID1_21)
                #print("%s: pedestrian %s pushes the button ID5_26 (waiting: %s)" %
                #      (traci.simulation.getTime(), ped, numWaiting_21))
                return True
    return False'''

def runSUMO(options=None):
    # Run in GUI or command line
    if options == "sumo-gui":
        sumoBinary = "sumo-gui"
    else:
        sumoBinary = "sumo"

    # start traci execution
    traci.start([sumoBinary, "-c", PATH + sumocfg, "--no-step-log", "true", "-W", "true"], label="master")
    run()

    return statisticxml()


# this is the main entry point of this script
if __name__ == "__main__":
    runRouteSampler(RS_get_options(cmdl=True))
    runSUMO("sumo-gui")
