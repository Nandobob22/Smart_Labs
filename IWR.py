import serial
import time
import numpy as np
from PySide6 import QtWidgets,QtCore  # Should work with PyQt5 / PySide2 / PySide6 as well
import pyqtgraph as pg
from PySide6.QtCore import QTimer  
from math import sqrt
from statistics import median

# Change the configuration file name
configFileName = 'profile.cfg'

CLIport = {}
Dataport = {}
byteBuffer = np.zeros(2**15,dtype = 'uint8')
byteBufferLength = 0;

Person_Count=0

min_y = 0.4
max_y = 3

min_x = -2
max_x = 2

cluster_radius=0.3

previous_frame=[]
current_frame=[]
point_frames=[]
raw_x=[]
raw_y=[]

person_count=0

data_set_count=0
data_filter_stop=5

# ------------------------------------------------------------------

# Function to configure the serial ports and send the data from
# the configuration file to the radar
def serialConfig(configFileName):
    
    global CLIport
    global Dataport
    # Open the serial ports for the configuration and the data ports
    
    # Raspberry pi
    #CLIport = serial.Serial('/dev/ttyACM0', 115200)
    #Dataport = serial.Serial('/dev/ttyACM1', 921600)
    
    # Mac
    CLIport = serial.Serial('/dev/tty.usbmodemR10310411', 115200)
    Dataport = serial.Serial('/dev/tty.usbmodemR10310414', 921600)
    
    # Windows
    #CLIport = serial.Serial('COM4', 115200)
    #Dataport = serial.Serial('COM3', 921600)


    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:
        CLIport.write((i+'\n').encode())
        print(i)
        time.sleep(0.01)
        
    return CLIport, Dataport

# ------------------------------------------------------------------

# Function to parse the data inside the configuration file
def parseConfigFile(configFileName):
    configParameters = {} # Initialize an empty dictionary to store the configuration parameters
    
    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:
        
        # Split the line
        splitWords = i.split(" ")
        
        # Hard code the number of antennas, change if other configuration is used
        numRxAnt = 4
        numTxAnt = 3
        
        # Get the information about the profile configuration
        if "profileCfg" in splitWords[0]:
            startFreq = int(float(splitWords[2]))
            idleTime = int(splitWords[3])
            rampEndTime = float(splitWords[5])
            freqSlopeConst = float(splitWords[8])
            numAdcSamples = int(splitWords[10])
            numAdcSamplesRoundTo2 = 1;
            
            while numAdcSamples > numAdcSamplesRoundTo2:
                numAdcSamplesRoundTo2 = numAdcSamplesRoundTo2 * 2;
                
            digOutSampleRate = int(splitWords[11]);
            
        # Get the information about the frame configuration    
        elif "frameCfg" in splitWords[0]:
            
            chirpStartIdx = int(float(splitWords[1]));
            chirpEndIdx = int(float(splitWords[2]));
            numLoops = int(float(splitWords[3]));
            numFrames = int(float(splitWords[4]));
            framePeriodicity = int(float(splitWords[5]));

            
    # Combine the read data to obtain the configuration parameters           
    numChirpsPerFrame = (chirpEndIdx - chirpStartIdx + 1) * numLoops
    configParameters["numDopplerBins"] = numChirpsPerFrame / numTxAnt
    configParameters["numRangeBins"] = numAdcSamplesRoundTo2
    configParameters["rangeResolutionMeters"] = (3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * numAdcSamples)
    configParameters["rangeIdxToMeters"] = (3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * configParameters["numRangeBins"])
    configParameters["dopplerResolutionMps"] = 3e8 / (2 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * configParameters["numDopplerBins"] * numTxAnt)
    configParameters["maxRange"] = (300 * 0.9 * digOutSampleRate)/(2 * freqSlopeConst * 1e3)
    configParameters["maxVelocity"] = 3e8 / (4 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * numTxAnt)
    
    return configParameters
   
# ------------------------------------------------------------------

# Funtion to read and parse the incoming data
def readAndParseData14xx(Dataport, configParameters):
    global byteBuffer, byteBufferLength
    
    # Constants
    OBJ_STRUCT_SIZE_BYTES = 12;
    BYTE_VEC_ACC_MAX_SIZE = 2**15;
    MMWDEMO_UART_MSG_DETECTED_POINTS = 1;
    MMWDEMO_UART_MSG_RANGE_PROFILE   = 2;
    maxBufferSize = 2**15;
    magicWord = [2, 1, 4, 3, 6, 5, 8, 7]
    
    # Initialize variables
    magicOK = 0 # Checks if magic number has been read
    dataOK = 0 # Checks if the data has been read correctly
    frameNumber = 0
    detObj = {}
    
    readBuffer = Dataport.read(Dataport.in_waiting)
    byteVec = np.frombuffer(readBuffer, dtype = 'uint8')
    byteCount = len(byteVec)
    
    # Check that the buffer is not full, and then add the data to the buffer
    if (byteBufferLength + byteCount) < maxBufferSize:
        byteBuffer[byteBufferLength:byteBufferLength + byteCount] = byteVec[:byteCount]
        byteBufferLength = byteBufferLength + byteCount
        
    # Check that the buffer has some data
    if byteBufferLength > 16:
        
        # Check for all possible locations of the magic word
        possibleLocs = np.where(byteBuffer == magicWord[0])[0]

        # Confirm that is the beginning of the magic word and store the index in startIdx
        startIdx = []
        for loc in possibleLocs:
            check = byteBuffer[loc:loc+8]
            if np.all(check == magicWord):
                startIdx.append(loc)
               
        # Check that startIdx is not empty
        if startIdx:
            
            # Remove the data before the first start index
            if startIdx[0] > 0 and startIdx[0] < byteBufferLength:
                byteBuffer[:byteBufferLength-startIdx[0]] = byteBuffer[startIdx[0]:byteBufferLength]
                byteBuffer[byteBufferLength-startIdx[0]:] = np.zeros(len(byteBuffer[byteBufferLength-startIdx[0]:]),dtype = 'uint8')
                byteBufferLength = byteBufferLength - startIdx[0]
                
            # Check that there have no errors with the byte buffer length
            if byteBufferLength < 0:
                byteBufferLength = 0
                
            # word array to convert 4 bytes to a 32 bit number
            word = [1, 2**8, 2**16, 2**24]
            
            # Read the total packet length
            totalPacketLen = np.matmul(byteBuffer[12:12+4],word)
            
            # Check that all the packet has been read
            if (byteBufferLength >= totalPacketLen) and (byteBufferLength != 0):
                magicOK = 1
    
    # If magicOK is equal to 1 then process the message
    if magicOK:
        # word array to convert 4 bytes to a 32 bit number
        word = [1, 2**8, 2**16, 2**24]

        
        # Initialize the pointer index
        idX = 0
        
        # Read the header
        magicNumber = byteBuffer[idX:idX+8]
        #print("magicNumber: "+str(byteBuffer[idX:idX+4])+" : "+str(magicNumber))
        idX += 8

        version = format(np.matmul(byteBuffer[idX:idX+4],word),'x')
        #print("version: "+str(byteBuffer[idX:idX+4])+" : "+str(version))
        idX += 4

        totalPacketLen = np.matmul(byteBuffer[idX:idX+4],word)
        #print("totalPacketLen: "+str(byteBuffer[idX:idX+4])+" : "+str(totalPacketLen))
        idX += 4

        platform = format(np.matmul(byteBuffer[idX:idX+4],word),'x')
        #print("platform: "+str(byteBuffer[idX:idX+4])+" : "+str(platform))
        idX += 4

        frameNumber = np.matmul(byteBuffer[idX:idX+4],word)
        #print("frameNumber: "+str(byteBuffer[idX:idX+4])+" : "+str(frameNumber))
        idX += 4

        timeCpuCycles = np.matmul(byteBuffer[idX:idX+4],word)
        #print("timeCpuCycles: "+str(byteBuffer[idX:idX+4])+" : "+str(timeCpuCycles))
        idX += 4

        numDetectedObj = np.matmul(byteBuffer[idX:idX+4],word)
        #print("numDetectedObj: "+str(byteBuffer[idX:idX+4])+" : "+str(numDetectedObj))
        idX += 4

        numTLVs = np.matmul(byteBuffer[idX:idX+4],word)
        #print("num TLVs: "+str(byteBuffer[idX:idX+4])+" : "+str(numTLVs))
        idX += 4
        
        #subFrameNumber = np.matmul(byteBuffer[idX:idX+4],word)
        #print("subFrameNumber: "+str(byteBuffer[idX:idX+4])+" : "+str(subFrameNumber))
        #idX += 4        
        
        #print()
        
        # Read the TLV messages
        for tlvIdx in range(numTLVs):
            
            # word array to convert 4 bytes to a 32 bit number
            word = [1, 2**8, 2**16, 2**24]

            # Check the header of the TLV message
            tlv_type = np.matmul(byteBuffer[idX:idX+4],word)
            #print("tlv_type: "+str(byteBuffer[idX:idX+4])+" : "+str(tlv_type))
            idX += 4

            tlv_length = np.matmul(byteBuffer[idX:idX+4],word)
            idX += 4
            
            # Read the data depending on the TLV message
            if tlv_type == MMWDEMO_UART_MSG_DETECTED_POINTS:
                            
                # word array to convert 4 bytes to a 16 bit number
                word = [1, 2**8]
                tlv_numObj = np.matmul(byteBuffer[idX:idX+2],word)
                idX += 2
                tlv_xyzQFormat = 2**np.matmul(byteBuffer[idX:idX+2],word)
                idX += 2
                
                # Initialize the arrays
                rangeIdx = np.zeros(tlv_numObj,dtype = 'int16')
                dopplerIdx = np.zeros(tlv_numObj,dtype = 'int16')
                peakVal = np.zeros(tlv_numObj,dtype = 'int16')
                x = np.zeros(tlv_numObj,dtype = 'int16')
                y = np.zeros(tlv_numObj,dtype = 'int16')
                z = np.zeros(tlv_numObj,dtype = 'int16')
                
                for objectNum in range(tlv_numObj):
                    
                    # Read the data for each object
                    rangeIdx[objectNum] =  np.matmul(byteBuffer[idX:idX+2],word)
                    idX += 2
                    dopplerIdx[objectNum] = np.matmul(byteBuffer[idX:idX+2],word)
                    idX += 2
                    peakVal[objectNum] = np.matmul(byteBuffer[idX:idX+2],word)
                    idX += 2
                    x[objectNum] = np.matmul(byteBuffer[idX:idX+2],word)
                    idX += 2
                    y[objectNum] = np.matmul(byteBuffer[idX:idX+2],word)
                    idX += 2
                    z[objectNum] = np.matmul(byteBuffer[idX:idX+2],word)
                    idX += 2
                    
                # Make the necessary corrections and calculate the rest of the data
                rangeVal = rangeIdx * configParameters["rangeIdxToMeters"]
                dopplerIdx[dopplerIdx > (configParameters["numDopplerBins"]/2 - 1)] = dopplerIdx[dopplerIdx > (configParameters["numDopplerBins"]/2 - 1)] - 65535
                dopplerVal = dopplerIdx * configParameters["dopplerResolutionMps"]
                #x[x > 32767] = x[x > 32767] - 65536
                #y[y > 32767] = y[y > 32767] - 65536
                #z[z > 32767] = z[z > 32767] - 65536
                x = x / tlv_xyzQFormat
                y = y / tlv_xyzQFormat
                z = z / tlv_xyzQFormat
                
                # Store the data in the detObj dictionary
                detObj = {"numObj": tlv_numObj, "rangeIdx": rangeIdx, "range": rangeVal, "dopplerIdx": dopplerIdx, \
                          "doppler": dopplerVal, "peakVal": peakVal, "x": x, "y": y, "z": z}
                
                dataOK = 1             
        
  
        # Remove already processed data
        if idX > 0 and byteBufferLength > idX:
            shiftSize = totalPacketLen
               
            byteBuffer[:byteBufferLength - shiftSize] = byteBuffer[shiftSize:byteBufferLength]
            byteBuffer[byteBufferLength - shiftSize:] = np.zeros(len(byteBuffer[byteBufferLength - shiftSize:]),dtype = 'uint8')
            byteBufferLength = byteBufferLength - shiftSize
            
            # Check that there are no errors with the buffer length
            if byteBufferLength < 0:
                byteBufferLength = 0
                

    return dataOK, frameNumber, detObj

# ------------------------------------------------------------------

class Point():
    def __init__(self,cord,group=-1):
        self.point=cord
        self.x=cord[0]
        self.y=cord[1]
        self.group=group
        self.used=-1

    def __eq__(self, other):
        if self.x==other.x and self.y==other.y:
            return True
        return False

    def __ne__(self, other):
        if self.x!=other.x or self.y!=other.y:
            return True
        return False

def CalculatePointDistance(Point1:Point, Point2:Point) -> float:
    x1=Point1.x
    y1=Point1.y

    x2=Point2.x
    y2=Point2.y

    distance=sqrt(((x2-x1)**2)+((y2-y1)**2))

    return distance

def CalculatePointCenter(Point1:Point, Point2:Point) -> Point:
    x1=Point1.x
    y1=Point1.y

    x2=Point2.x
    y2=Point2.y

    center_x=(x1+x2)/2
    center_y=(y1+y2)/2

    center=Point((center_x,center_y))

    return center

def CalculatePointCluster(x:list, y:list) -> dict:
    #Format data as (x1,y1),(x2,y2),...
    xy=zip(x,y)

    #Create list of points that are within specified bound
    bound_xy = []
    for point in xy:
        current_x=point[0]
        current_y=point[1]
        if current_x > min_x and current_x < max_x: #If within x bounds
            if current_y > min_y and current_y < max_y: #If within y bounds
                bound_xy.append(Point(point))

    if bound_xy: #If we have any points in the specified bound
        #Loop through all the points generate sweeping average points based the clustering radius
        clustered_xy=list(bound_xy)
        all_points_clustered=False
        while not all_points_clustered:
            for point_1 in clustered_xy:
                for point_2 in clustered_xy:
                    if point_1 != point_2:
                        if CalculatePointDistance(point_1,point_2) <= cluster_radius:
                            #Remove two point from clustered_xy
                            del clustered_xy[clustered_xy.index(point_1)]
                            del clustered_xy[clustered_xy.index(point_2)]

                            #add center point to clustered_xy
                            clustered_point=CalculatePointCenter(point_1,point_2)
                            clustered_xy.append(clustered_point)

                            #break
                            break
                else:
                    continue #Continue to next point_2 if did not break
                break #Break from main for loop if inner loop was breaked
            else:
                all_points_clustered=True

        #Calculate the point density for all the groups
        group_cluster_density=[]
        for index,temp_clustered_point in enumerate(clustered_xy):
            group_cluster_density.append(0)
            for temp_bound_point in bound_xy:
                if CalculatePointDistance(temp_clustered_point,temp_bound_point) <= cluster_radius:
                    group_cluster_density[index]+=1
        group_cluster_median=median(group_cluster_density)
        #print(f"Group Point Density: {group_cluster_density}")
        #print(f"Frame Median Density: {group_cluster_median}")

        #Remove all groups with lower point density than the density median
        clustered_xy_copy=list(clustered_xy)
        group_cluster_density_copy=list(group_cluster_density)
        clustered_xy=[]
        group_cluster_density=[]
        for index,dense_point in enumerate(clustered_xy_copy):
            if group_cluster_density_copy[index]>=group_cluster_median:
                clustered_xy.append(dense_point)
                group_cluster_density.append(group_cluster_density_copy[index])

        #Remove any overlapping clusters with the lower denisty
        density_fight_end=False
        while not density_fight_end:
            for index,dense_point in enumerate(clustered_xy):
                for index_1,dense_point_1 in enumerate(clustered_xy):
                    if dense_point != dense_point_1:
                        if CalculatePointDistance(dense_point,dense_point_1) < cluster_radius*2:
                            if group_cluster_density[index] > group_cluster_density[index_1]:
                                del clustered_xy[index_1]
                                del group_cluster_density[index_1]
                            else:
                                del clustered_xy[index]
                                del group_cluster_density[index]
                            break
                else:
                    continue
                break
            else:
                density_fight_end=True

        #Format clustered points for plotting
        clustered_x=[p.x for p in clustered_xy]
        clustered_y=[p.y for p in clustered_xy]

        #print("clustered_xy:")
        #for point in clustered_xy:
            #print(f"\t({point.x},{point.y})")

        #Format bound points for plotting
        bound_x=[p.x for p in bound_xy]
        bound_y=[p.y for p in bound_xy]

        data_export={
            "clustered":{
                "x":clustered_x,
                "y":clustered_y
            },
            "bound":{
                "x":bound_x,
                "y":bound_y
            }
        }

        return data_export
    return {}

        #print("bound_xy:")
        #for point in bound_xy:
            #print(f"\t({point.x},{point.y})")

# Funtion to update the data and display in the plot
def update() -> None:
     
    dataOk = 0
    global detObj
    global person_count
    global point_frames
    global reset_count
    global CLIport, Dataport
    global current_frame,previous_frame
    global data_set_count
    global raw_x,raw_y
    x = []
    y = []

    # Read and parse the received data
    try:
        dataOk, frameNumber, detObj = readAndParseData14xx(Dataport, configParameters)
    except:
        pass

    if dataOk and len(detObj["x"]) > 0:
        #Get Raw x and y data
        x = -detObj["x"]
        y = detObj["y"]

        #Calculate clusters from raw data
        data_set=CalculatePointCluster(x,y)

        #If any clusters found
        if data_set:

            #Clear plot frame
            plot.clear()

            #Draw rectangle to show sensor bounds
            bounding_rect=pg.RectROI(pos=[min_x,min_y],size=[max_x-min_x,max_y-min_y],pen=pg.mkPen('w',width=1))
            plot.addItem(bounding_rect)

            #Draw line to show person count trigger
            line_border=pg.LineROI(pos1=[0,min_y],pos2=[0,max_y],width=0,pen=pg.mkPen('b',width=1))
            plot.addItem(line_border)

            #Plot the unclustered data points
            bound_x=data_set["bound"]["x"]
            bound_y=data_set["bound"]["y"]
            plot.plot(bound_x, bound_y, pen=None, symbol='o',symbolPen="Red",symbolBrush="Red")
            
            #Plot the clustered points with a circle representing the clustering radius
            clustered_x=data_set["clustered"]["x"]
            clustered_y=data_set["clustered"]["y"]
            plot.plot(clustered_x, clustered_y, pen=None, symbol='+',symbolPen="White",symbolBrush="White")
            for index,point in enumerate(zip(clustered_x,clustered_y)):
                point_radius_circle=pg.CircleROI(pos=[point[0]-cluster_radius,point[1]-cluster_radius], radius=cluster_radius, pen=pg.mkPen('w',width=1))
                plot.addItem(point_radius_circle)


            #Get all clustered points from the current frame of data
            current_frame=[Point(point,index) for index,point in enumerate(zip(clustered_x,clustered_y))]
            
            '''
            point_frames.append(current_frame)
            if len(point_frames) == data_set_count:
                point_frame_groups={}
                for current_index in range(len(max(point_frames))):
                    point_frame_groups[current_index]=[]
                for frame in point_frames:
                    frame_point={obj.group: obj for obj in frame}
                    for key in frame_point:
                        point_frame_groups[key].append(frame_point[key])
            #'''

            #if previous_frame:
            for previous_point in previous_frame:
                point_distances=[]
                for current_point in current_frame:
                    point_distances.append(CalculatePointDistance(previous_point,current_point))
                if min(point_distances) <= cluster_radius:
                    closest_point_index=point_distances.index(min(point_distances))
                    if current_frame[closest_point_index].used==-1:
                        current_frame[closest_point_index].used=1
                        next_point=current_frame[point_distances.index(min(point_distances))]
                        next_point.group=previous_point.group
                        point_text=pg.TextItem(f"Person: {next_point.group}")
                        plot.addItem(point_text)
                        point_text.setPos(next_point.x-cluster_radius,next_point.y-cluster_radius)
                        if next_point.x >= 0 and previous_point.x < 0:
                            person_count+=1
                        elif next_point.x <0 and previous_point.x >=0:
                            person_count-=1

                    person_count_label.setText(f"Person Count: {person_count}")

            previous_frame=current_frame

        '''
        if data_set:
            plot.clear()

            bound_x=data_set["bound"]["x"]
            bound_y=data_set["bound"]["y"]
            plot.plot(bound_x, bound_y, pen=None, symbol='o',symbolPen="Red",symbolBrush="Red")
            
            clustered_x=data_set["clustered"]["x"]
            clustered_y=data_set["clustered"]["y"]
            plot.plot(clustered_x, clustered_y, pen=None, symbol='+',symbolPen="White",symbolBrush="White")
            for point in zip(clustered_x,clustered_y):
                point_radius_circle=pg.CircleROI(pos=[point[0]-cluster_radius,point[1]-cluster_radius], radius=cluster_radius, pen=pg.mkPen('w',width=1))
                plot.addItem(point_radius_circle)
        '''

# -------------------------    MAIN   -----------------------------------------  

time.sleep(5)

# Configurate the serial port
CLIport, Dataport = serialConfig(configFileName)

# Get the configuration parameters from the configuration file
configParameters = parseConfigFile(configFileName)

app = QtWidgets.QApplication([])
app.setStyleSheet("QLabel{font-size: 24pt;}")

## Define a top-level widget to hold everything
w = QtWidgets.QWidget()
w.setWindowTitle('IWR Output')

timer = QTimer(w)        
timer.timeout.connect(update)        
timer.start(5) #5 miliseconds = 200Hz sampling rate

## Create plot to be placed inside
plot = pg.PlotWidget()
plot.viewport().setAttribute(QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, False)
plot.setAspectLocked()
plot.setXRange(min_x-0.2, max_x+0.2, padding=0)
plot.setYRange(min_y-0.2,max_y+0.2,padding=0)
plot.setLabel('left', "Forward Distance from sensor (m)")
plot.setLabel('bottom', "Horizotnal Distance from sensor (m)")
## Create a grid layout to manage the widgets size and position
layout = QtWidgets.QVBoxLayout()
w.setLayout(layout)

person_count_label=QtWidgets.QLabel("Person Count: 0")
layout.addWidget(person_count_label)

layout.addWidget(plot)  # Add plot to window layout
## Display the widget as a new window
w.show()

## Start the Qt event loop
app.exec()

