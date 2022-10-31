
----- DEMONSTRATION VERSION ------  
For demonstration of the pre-selection tool, run run_test_program.py  
----- DEMONSTRATION VERSION ------

Person responsible: Sonja Grönroos sonja.gronroos@cern.ch

The repository contains the software that can be used to scan and evaluate an 8inch HGCAL silicon sensor.
By creating a new scan mapping, also other sensor geometries could be inspected.
The evaluation is done by an ensemble of deep CNNs (referred to as the pre-selection, or PS). 
The images produced by the scan program are analysed by the PS, and those images that are flagged to potentially 
contain anomalies are indicated to the inspector so that the anomalous area can be inspected and potentially cleaned. 

The accuracy of the current model is 0.95. The accuracy is expected to increase as the model is trained further with new data. 
This is why all images are still saved in compressed jpeg form.  
In the future, images reliably determined to be normal by the PS could be removed and only those containing anomalies would 
be saved for future use in re-training the model. 
Only the anomalous images need to be saved for re-training purposes as data is heavily imbalanced and lacks in anomalous areas.

### Additional resources
Video demonstration of a scan of a (very dirty) sensor:  
https://youtu.be/5Pm26gaBaMA

README style description of the key programs in the software:  
doc/software_description.txt:

Compressed version of run instructions:  
doc/HexScan_run_commands.txt:  


## How to scan a sensor

1. Open anaconda prompt on the GPU PC


2. Navigate to the correct directory  
C:\Users\HexaScan_AI_user\Desktop\HexaScan_withAI


3. If not already activated, activate conda environment  
conda activate HEXASCAN_AI


4. Prepare to scan a sensor. See instructions in document "Procedure_updated" if full procedure is unclear.

### Parallel scan + pre-selection

5. Run parallel scan and pre-selection with the command  
python run_photo_taking_with_PS.py --CampaignName [insert here]

the CampaignName argument is mandatory to give for the command.   
There are, in addition, six more arguments, which are optional:  
--DUTName : [give name of DUT]  
--SensorGeometry : [HPK_198ch_8inch, HPK_432ch_8inch, custom_map_LD_396, custom_map_HD_396, custom_map_LD_385 (default), custom_map_HD_385]  
--PS : [1: use pre-selection (default), 0: no pre-selection]  
--COMPORT : [COM port of the xy-stage, default = 3]  
--Threshold : [Classification threshold, default = 0.1]  
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
--SensorGeometry : [HPK_198ch_8inch, HPK_432ch_8inch, custom_map_LD_396, custom_map_HD_396, custom_map_LD_385 (default), custom_map_HD_385]  
--CampaignName : give name of measurement campaign given in previous run  
--DUTName : give name of DUT determined in previous run  
--N_images : number of images in scan, see how many images were produced by the scan (depends on scan map), ref. Table 1.  

and optional:  
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


Table 1. The scan mappings

| Name of scan map  | DetailLevel | N_images |
|-------------------|-------------|----------|
| HPK_198ch_8inch   | 0           | 192      |
|                   | 1           | 368      |
|                   | 2           | 529      |
| HPK_432ch_8inch   | 0           | 432      |
|                   | 1           | 864      |
|                   | 2           | 1273     |
| custom_map_xx_396 | 0           | 396      |
| custom_map_xx_385 | 0           | 385      |


## How to use the validation tool

This tool can be used to
1. VALIDATE the pre-selected anomalous images and the saved fraction of normal images.
Useful when the inspector has no time to validate images straight after scan but has motivation to do it later.
2. SHOW a summary for the pre-selected annotations compared to the human-validated ones: can be used to monitor the accuracy of the pre-selection.


Run  
python validation_tool.py  
with mandatory arguments  
--mode : [VALIDATE or SHOW]  
--CampaignName : [name of campaign]  
and optional argument  
--DUTName : [name of DUT]

if you do not specify the DUTName, in validation mode you will be suggested the DUTs that have not yet been validated.
In show mode all DUTs of the campaign will be printed.
