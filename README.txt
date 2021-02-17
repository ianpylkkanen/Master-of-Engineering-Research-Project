Ian Pylkkanen
Master of Engineering Project

Rensselaer Polytechnic Institute
Department of Mechanical, Aerospace, and Nuclear engineering
Intelligent Strucutral Systems Laboratory (ISSL)
Troy, NY
Advisor: Fotis Kopsaftopoulos

January 2020 - December 2020
-----------------------------------------------------------------------------------------------
This folder contains all the contents of my resesarch project regarding the simulation of 
structural health monitoring (SHM) of a 1U CubeSat.
-----------------------------------------------------------------------------------------------
Files Contained:
-----------------------------------------------------------------------------------------------
Final Presentation

PDF: Pylkkanen_FinalPresentation_MEng2020.pdf
-----------------------------------------------------------------------------------------------
NX

Main Files
Part File: CubeSatmm.prt
FEM File:  CubeSatmm_fem1.fem
SIM File:  CubeSatmm_sim1.sim

Data Files
AFU Files: CubeSatmm_sim1.afu

Mesh Convergence Files: cubesatmm_sim1-solution_1_dynamics_mesh_conv.dat
                        cubesatmm_sim1-solution_1_dynamics_mesh_conv.diag
                        cubesatmm_sim1-solution_1_dynamics_mesh_conv.f04
                        cubesatmm_sim1-solution_1_dynamics_mesh_conv.f06
                        cubesatmm_sim1-solution_1_dynamics_mesh_conv.txt
                        cubesatmm_sim1-solution_1_dynamics_mesh_conv.op2

Static Load Files:      cubesatmm_sim1-solution_1_static_load.dat
                        cubesatmm_sim1-solution_1_static_load.diag
                        cubesatmm_sim1-solution_1_static_load.f04
                        cubesatmm_sim1-solution_1_static_load.f06
                        cubesatmm_sim1-solution_1_static_load.txt
                        cubesatmm_sim1-solution_1_static_load.op2

Dynamics:               cubesatmm_sim1-response_dynamics_2-event_2.afu
                        cubesatmm_sim1-response_dynamics_2-event_2.eef
-----------------------------------------------------------------------------------------------
MATLAB

System ID Script: system_identification_demo_2 .m - written by Kopsaftopoulos, F.
Data Load: Signal_Data.m
Sensor Responses: sensor1_resp.mat
                  sensor2_resp.mat
                  sensor3_resp.mat
                  sensor4_resp.mat
                  sensor5_resp.mat
                  sensor6_resp.mat
MAT File for System ID Call: signals.mat  - NOTE: This mat file will change depending on which
                                                  sensor is being analyzed.  Change the sensor
                                                  in the Signal_Data.m file.
-----------------------------------------------------------------------------------------------
CSV

Excitaion Signal: excitation_signal.csv - NOTE: This file is only for import into NX and load
                                                in Signal_Data.m file.
                                                The signal was generated in Signal_Data.m file
                                                and should not be edited.
-----------------------------------------------------------------------------------------------
For any question please contact me at: ian.pylkkanen@gmail.com