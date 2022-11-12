This is an updated version of the software in https://gitlab.cern.ch/CLICdp/HGCAL/hexascan by Thorben Quast.

Person responsible: Sonja Grönroos sonja.gronroos@cern.ch

The repository contains the software that can be used to scan and evaluate an 8inch HGCAL silicon sensor and partials.
By creating a new scan mapping, also other sensor geometries could be inspected.
The evaluation is done by an ensemble of deep CNNs (referred to as the pre-selection, or PS). 
The images produced by the scan program are analysed by the PS, and those images that are flagged to potentially 
contain anomalies are indicated to the inspector so that the anomalous area can be inspected and potentially cleaned. 

The accuracy of the current model is ~0.95. The accuracy is expected to increase as the model is trained further with new data. 
This is why all images are still saved in compressed jpeg form.  
In the future, images reliably determined to be normal by the PS could be removed and only those containing anomalies would 
be saved for future use in re-training the model. 
Only the anomalous images need to be saved for re-training purposes as data is heavily imbalanced and lacks in anomalous areas.

### Additional resources
Video demonstration of a scan of a (very dirty) sensor:  
[Link to Youtube](https://youtu.be/5Pm26gaBaMA)

README style description of the key programs in the software:  
[doc/software_description.txt](doc/software_description.txt):

Compressed version of run instructions:  
[doc/HexScan_run_commands.txt](doc/HexScan_run_commands.txt):  

Requirements and dependencies:  
Python 3.10 + [requirements.txt](requirements.txt )  

Dummy program to demonstrate pre-selection on 20 images (CPU-friendly):  
run  
python run_test_program.py  


## How to scan a sensor

1. Open anaconda prompt on the GPU PC


2. Navigate to the correct directory  
C:\Users\HexaScan_AI_user\Desktop\HexaScan_withAI


3. If not already activated, activate conda environment  
conda activate HEXASCAN_AI


4. Prepare to scan a sensor. See instructions in document [Procedure_updated](doc/Procedure_updated.docx) if full procedure is unclear.

### Parallel scan + pre-selection

5. Run parallel scan and pre-selection with the command  
python run_photo_taking_with_PS.py --CampaignName [insert here]

the CampaignName argument is mandatory to give for the command.   
There are, in addition, six more arguments, which are optional:  
--DUTName : [give name of DUT]  
--SensorGeometry : [HPK_198ch_8inch, HPK_432ch_8inch, custom_map_LD_Full_396, etc.]  
--PS : [1: use pre-selection (default), 0: no pre-selection]  
--COMPORT : [COM port of the xy-stage, default = 3]  
--ClassifierDate : date of classifier model (CNNs/class_cnn_xxx), in format YYYY-MM-DD  
--Threshold : [Classification threshold, defaults in Table 2]  
--Grid : [1: use only default grid (default), 2: use also secondary grid]   

6. As the terminal will prompt you to, perform the sensor alignment, test image and scratch pad reading. Then initiate scan.


7. Let the scan finish. The images are analysed in parallel by the pre-selection algorithm as the scan progresses.
All images are saved as .jpeg, and the images found to be anomalous by the PS are saved as both .png and .npy files in the output directory.
A fraction of images deemed normal is also saved.
On the scan map in Figure 1, you can follow how the scan progresses.
The scan takes approx. 15 minutes to finish.


8. After scan is finished, you are asked to validate the annotated images. 
You will also validate 10 % of the normal images.
You can skip this step by writing "no" in the command prompt when asked if you want to validate.
Or you can exit validation later by typing END when prompted.


9. The program prints the scan indices where anomalies were detected by the PS OR where you annotated to be anomalies in the normal images during validation.
You are then prompted to go through the indices.
If no indices were selected, you can input one.
The xy-stage will move to the position corresponding to a selected scan index for inspection. 
Live footage of the inspected scan position can be viewed on screen. 
After you quit the inspection/cleaning, a rescan photo is taken and saved in the /jpegs folder for future reference.


10.  End of program: inspection of the guard ring.
With certain scan mappings you will have the possibility to go through the guard ring images on screen, and then move the xy-stage to a position corresponding to the image.
Or you can perform the guard ring inspection manually with the joystick.

### Separate scan + pre-selection

5. Run the scan with no pre-selection with the command  
python run_photo_taking_with_PS.py --CampaignName [insert here] --PS 0  

This program will implement the picture-taking scan, without any pre-selection: all images will be saved as .npy and .jpeg.


4. After scan has finished, exit previous program and run command   
python run_PS.py  
with mandatory arguments:  
--SensorGeometry : [HPK_198ch_8inch, HPK_432ch_8inch, custom_map_LD_Full_396, etc.]  
--CampaignName : give name of measurement campaign given in previous run  
--DUTName : give name of DUT determined in previous run  
--N_images : number of images in scan, see how many images were produced by the scan (depends on scan map), ref. Table 1.  

and optional:  
--ClassifierDate : date of classifier model (CNNs/class_cnn_xxx), in format YYYY-MM-DD  
--Threshold : [no need to change, default should be preferred]    
--Grid : [1: use only default grid (default), 2: use also secondary grid]  
--Verbose :  default 0 [change to 1 if you want verbosity]    

This program will run the PS through the images produced by the scan program and remove 90 % of the normal images.

5. Run  
python run_cleaning.py  
with arguments  
--SensorGeometry : sensor geometry, give same as before  
--CampaignName : give name of measurement campaign  
--DUTName : give name of DUT  
--DetailLevel : OPTIONAL default 2 [with custom maps use 0], ref. Table 1.  
--COMPORT : OPTIONAL default 3 [no need to change if no error]


Table 1. The scan mappings.

| Name of scan map       | DetailLevel | Alignment cells<br/>LD | Alignment cells <br/>HD | N_images  |
|------------------------|-------------|-----------------------|-------------------------|-----------|
| HPK_198ch_8inch        | 0           | 103 and 104           | -                       | 192       |
|                        | 1           |                       |                         | 368       |
|                        | 2           |                       |                         | 529       |
| HPK_432ch_8inch        | 0           | -                     | 228 and 229             | 432       |
|                        | 1           |                       |                         | 864       |
|                        | 2           |                       |                         | 1273      |
| custom_map_xD_Full_396 | 0           | 103 and 104           | 228 and 229             | 396       |
| custom_map_xD_Full_385 | 0           | 103 and 104           | 228 and 229             | 385       |
| custom_map_xD_Top      | 0           |                       |                         | up to 385 |
| custom_map_xD_Bottom   | 0           |                       |                         |           |
| custom_map_xD_Left     | 0           |                       |                         |           |
| custom_map_xD_Right    | 0           |                       |                         |           |
| custom_map_xD_Five     | 0           |                       | -                       |           |
| custom_map_xD_Three    | 0           |                       |                         |           |


Table 2. The classifier CNNs and thresholds.

| Name of classifier CNN                 | ClassifierDate | Threshold |
|----------------------------------------|----------------|-----------|
| class_cnn_clean_fl_epoch_18_2022-11-03 | 2022-11-03    | 0.3       |
| class_cnn_clean_fl_epoch_20_2022-07-26 | 2022-07-26     | 0.1       |

### How to use the validation tool

This tool can be used to
1. VALIDATE the pre-selected anomalous images and the saved fraction of normal images.
Useful when the inspector has no time to validate images straight after scan but has motivation to do it later.
2. SHOW a summary for the pre-selected annotations compared to the human-validated ones: can be used to monitor the accuracy of the pre-selection.


Run  
python validation_tool.py  
with mandatory arguments  
--Mode : [VALIDATE or SHOW]  
--CampaignName : [name of campaign]  
and optional argument  
--DUTName : [name of DUT]

if you do not specify the DUTName, in validation mode you will be suggested the DUTs that have not yet been validated.
In show mode all DUTs of the campaign will be printed.


### How to use the retraining tool (in development)

Run  
python retrain_tool.py 
with mandatory arguments  
--Mode : [train, continue or test]  
--ModelName : name of the model to process in folder /CNNs, e.g. class_cnn_clean_fl_epoch_18_2022-11-03  

mode argument determines the action of the retrain tool:
1. Train: you will retrain the model architecture from scratch, with re-initialized weights. Usually only if you have new training data.
2. Continue : you can continue training the model. Weights will not be re-initialized. With old or new training data.
3. Test : you will test the model. With old or new test data.

Several parameters affect the training process. 
For each model, these can be accessed in trainParams.json files in 
folder [class_cnn_test_outputs](CNNs/class_cnn_test_outputs).
The test output figures for models are also saved here.


### How to generate scan map files

Run  
python create_scan_map.py  
with mandatory arguments  
--SensorType [HD or LD]   
--Partial [Three, Five, Bottom, Top, Full, Left, Right]  

You might make changes to some map parameters to generate a map of your wishes. 
Parameters on lines 61 - 81 determine the extremities and the increments of the scan.
