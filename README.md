# LiquidMetal-RNN-Monitoring
RNN-based anomaly detection for Liquid Metal Reactor heater zones. Developed during Ph.D. research at Purdue University.

**Overview**
This repository implements a specialized monitoring framework for GenIV Liquid Metal Reactors (LMR). It utilizes data from an experimental liquid sodium facility to benchmark various Recurrent Neural Network (RNN) architectures in detecting anomalies in temperature sensor signals. As part of my Ph.D. research at Purdue University, this project bridges the gap between high-fidelity nuclear thermal hydraulics and deep learning. 

**Technical Approach**
-Data Source: Experimental time-series data from liquid sodium heater zones.
-Heuristic Labeling: Ground truth is established using a physics-informed 5th and 95th percentile thresholding strategy to isolate significant thermal excursions.
-Architectures: Comparative benchmarking of LSTM, GRU, Bi-LSTM and Bi-GRU models.
-Detection Logic: Employs EWMA-smoothed residuals and dynamic IQR-filtered thresholding to differentiate true anomalies from sensor noise.

**Repository Structure**
-generate_labels.py: Pre-processing script that applies engineering heuristics to raw data.

-main.py: The central execution engine for training, inference and performance evaluation.

-src/: Modular library containing model architectures, data loaders and signal processing utilities.

-data/: Directory for raw METL .mat files and generated labels.

**Steps**
1. Install dependencies from requirements.txt
2. Generate Ground Truth: Run the labeling script to create the anomaly labels based on the percentile thresholds
3. Train and Evaluate: Execute the main pipeline to benchmark the models
   
