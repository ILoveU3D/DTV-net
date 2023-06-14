import yaml
import os
y = yaml.load(open(r"./options.yaml"), yaml.FullLoader)

trainPath = os.path.join(y["dataPath"], y["subDir"]["train"])
validPath = os.path.join(y["dataPath"], y["subDir"]["valid"])
inputTrainData = os.path.join(y["dataPath"], y["subDir"]["trainInput"])
inputValidData = os.path.join(y["dataPath"], y["subDir"]["validInput"])
outputPath = os.path.join(y["dataPath"], "r.raw")
pretrain3 = y["pretrain3"]
checkpointPath = y["checkpointPath"]
debugPath = y["debug"]
standardAngleNum = y["standardAngleNum"]
standardDetectorSize = y["standardDetectorSize"]
standardVolumeSize = y["standardVolumeSize"]
standardSID = y["standardSID"]
standardSDD = y["standardSDD"]
beijingAngleNum = y["beijingAngleNum"]
beijingPlanes = y["beijingPlanes"]
beijingSubDetectorSize = y["beijingSubDetectorSize"]
beijingVolumeSize = y["beijingVolumeSize"]
beijingParameterRoot = y["beijingParameterRoot"]
beijingSID = y["beijingSID"]
beijingSDD = y["beijingSDD"]
sampleInterval = y["sampleInterval"]
