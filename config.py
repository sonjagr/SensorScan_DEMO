#relative path to the DLL library file for interacting to the Prior Scientific ProScan III controller and the H116 stage
PRIOR_STAGE_DLL = "external/PriorSDK_1_4_1/x64/PriorScientificSDK.dll"

#number of rows of pixels of the picture
#PICTURESIZE_Y = 2720
PICTURESIZE_Y = 2736 #old picture size

#number of columns of pixels of the picture
PICTURESIZE_X = 3840

#READOUT timeout after grabbing of an image has started (in microseconds)
RDOUTTIMEOUT = 1000

#for AI-based feature detection only
#dimensionality of the patching grid (=dimensionality of the fully-encoded image)
REDUCED_DIMENSION = (17, 24)
PATCHES = 408
PATCHSIZE = 160

DPI = 600
EIGHTBITMAX = 255

## paths for testing purposes
#TRAIN_DIR_LOC = r'C:\Users\sgroenro\PycharmProjects\hexascan_withai_inLab\datasets'
#MODELS_DIR_LOC = r'C:\Users\sgroenro\PycharmProjects\hexascan_withai_inLab\CNNs'
#IMAGES_DIR_LOC = r'F:\ScratchDetection\MeasurementCampaigns'
#VAL_DB_LOC = r'C:\Users\sgroenro\PycharmProjects\hexascan_withai_inLab\db_testing\validation_DB_testing'

## paths for GPUPC
TRAIN_DIR_LOC = r'C:\Users\HexaScan_AI_user\Desktop\HexaScan_withAI\datasets'
MODELS_DIR_LOC = r'C:\Users\HexaScan_AI_user\Desktop\HexaScan_withAI\CNNs'
IMAGES_DIR_LOC = r'D:\MeasurementCampaigns'
VAL_DB_LOC = r'C:\Users\HexaScan_AI_user\Desktop\HexaScan_withAI\db_testing\validation_DB_testing'
