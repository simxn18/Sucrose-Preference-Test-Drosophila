# Sucrose-Preference-Test-Drosophila
This repository contains analysis pipelines used for quantifying size preference in the Sucrose Preference Test (SPT), adapted for *Drosophila melanogaster*.
## Background
SPT is used as a proxy of anhedonic-like behavior in *Drosophila*. In short, flies are allowed to choose between a size with water and another with 5% sucrose. A lower preference for sucrose size is an anhedonic-like behavior.
# Data adquisition
Flies are recorded in a multi-way chamber, one fly per arena. The extremes of the arena contain cotton strips with water or 5% sucrose.
# Analysis steps
1. Videos are analyzed to track centroids of moving flies in the arena using a custom Bonsai.rx workflow (see video), contained in the /bonsai folder. The workflow also generates a CSV file containing the arena's distance in pixels.
2. Analysis is performed using a custom Python script. The analyzer works by reading CSV files in folders organized as follows: Main_folder/Condition/N/data.csv. Also, for each video, you need a config.csv file that specifies the side where sugar is applied: "right" or "left".
3. The results include time at each extreme and preference index.
## Status
The repository is under active development
## Author: Simón Guerra-Ayala
Please refer to this repository if it's used
