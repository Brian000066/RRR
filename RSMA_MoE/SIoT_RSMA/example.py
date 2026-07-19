import numpy as np
from input_settings import SIoT

def get_example_siot():
    # siot0=SIoT(0)
    # siot0.locations = ["L_2", "L_3", "L_4"]
    # siot0.neighbors = {1}
    # siot0.gain = 1
    # siot0.distance = 100
    # siot0.angle = 5
    siot0=SIoT(0)
    siot0.locations = ["L_1","L_2", "L_3", "L_4"]
    siot0.neighbors = {3}
    siot0.gain = 1
    siot0.distance = 100
    siot0.angle = 5
    # siot0=SIoT(0)
    # siot0.locations = ["L_1","L_2", "L_3"]
    # siot0.neighbors = {3}
    # siot0.gain = 1
    # siot0.distance = 100
    # siot0.angle = 5

    siot1 = SIoT(1)
    siot1.locations = ["L_3", "L_4","L_5"]
    siot1.neighbors = {2,3,4}
    siot1.gain = 1
    siot1.distance = 50
    siot1.angle = 15
    # siot1 = SIoT(1)
    # siot1.locations = ["L_3", "L_7","L_5"]
    # siot1.neighbors = {2,3,4}
    # siot1.gain = 1
    # siot1.distance = 50
    # siot1.angle = 15
    # siot1 = SIoT(1)
    # siot1.locations = ["L_7","L_5"]
    # siot1.neighbors = {2,3,4}
    # siot1.gain = 1
    # siot1.distance = 50
    # siot1.angle = 15
    
    siot2 = SIoT(2)
    siot2.locations = ["L_6", "L_7"]
    siot2.neighbors = {1}
    siot2.gain = 2
    siot2.distance = 80
    siot2.angle = 30
    

    # siot3 = SIoT(3)
    # siot3.locations = ["L_4", "L_6"]
    # siot3.neighbors = {1,4}
    # siot3.gain = 2
    # siot3.distance = 80
    # siot3.angle = 40
    siot3 = SIoT(3)
    siot3.locations = ["L_2","L_4", "L_6"]
    siot3.neighbors = {1,4}
    siot3.gain = 1
    siot3.distance = 100
    siot3.angle = 40
    # siot3 = SIoT(3)
    # siot3.locations = ["L_2", "L_6"]
    # siot3.neighbors = {0,1,4}
    # siot3.gain = 1
    # siot3.distance = 100
    # siot3.angle = 40
    

    siot4 = SIoT(4)
    siot4.locations = ["L_6", "L_7", "L_8"]
    siot4.neighbors = {1,3,5}
    siot4.gain = 2
    siot4.distance = 80
    siot4.angle = 45
    

    siot5 = SIoT(5)
    siot5.locations = ["L_8", "L_9"]
    siot5.neighbors = {4,6}
    siot5.gain = 1
    siot5.distance = 90
    siot5.angle = 60
    

    siot6 = SIoT(6)
    siot6.locations = ["L_10", "L_11", "L_12", "L_13"]
    siot6.neighbors = {5}
    siot6.gain = 1
    siot6.distance = 70
    siot6.angle = 70
    

    siot7 = SIoT(7)
    siot7.locations = ["L_9", "L_12", "L_13"]
    siot7.neighbors = {9}
    siot7.gain = 1.5
    siot7.distance = 80
    siot7.angle = 75
    

    siot8 = SIoT(8)
    siot8.locations = ["L_12", "L_13", "L_14"]
    siot8.neighbors = {9}
    siot8.gain = 1.5
    siot8.distance = 80
    siot8.angle = 75
    

    siot9 = SIoT(9)
    siot9.locations = ["L_9", "L_11", "L_12", "L_13", "L_14"]
    siot9.neighbors = {7,8}
    siot9.gain = 0.5
    siot9.distance = 95
    siot9.angle = 80

    request_locations=["L_1","L_2","L_3","L_4","L_5","L_6","L_7","L_8","L_9","L_10", "L_11", "L_12", "L_13", "L_14"]

    siots=[siot0,siot1,siot2,siot3,siot4,siot5,siot6,siot7,siot8,siot9]
    return siots, request_locations