import random
import numpy as np
from example import get_example_siot

def phase1(siots, request_locations):
    #Selected SIoT Set
    S=set() 
    
    #Get siot information
    #siots, num_locations_total , universal_locations= get_siot() 
    #for siot in siots:
    #    print(f"SIoT {siot.index}: Locations: {siot.locations}, Num_locations:{siot.num_locations}, Neighbors: {siot.get_neighbors()}, Distance: {siot.distance:.2f}, Angle: {siot.angle:.2f}, Gain: {siot.gain:.10f}")


    #Get request
    #request_locations = random.sample(universal_locations, random.randint(10, num_locations_total)) 
    #print("Request Locations:", request_locations)

    #test example
    #siots, request_locations=get_example_siot()
    
    


    
    #CCM the first SIoT selection
    siot_set, max_num_locations=compute_ccm(siots,S, request_locations)
    
    #SPCCM select SIoTs until the request satisfied
    siot_set=compute_spccm(siots, siot_set, request_locations,max_num_locations)
    #print("Phase1 Selected SIoTs:", [siot.index for siot in siot_set])

    unselected_siot_set = [siot for siot in siots if siot not in siot_set]
    

    return siot_set, unselected_siot_set
    

def compute_ccm(siots,S, request_locations):
    max_ccm=0
    max_index=None


    # Compute coverage values for each SIoT
    num_locations_values = [len(set(siot.locations) & set(request_locations)) for siot in siots]
    gains_values = [abs(siot.gain) for siot in siots]
    # Find the max and min
    max_num_locations = max(num_locations_values)
    min_num_locations = min(num_locations_values)
    max_gain=max(gains_values)
    min_gain=min(gains_values)
    
    #print(f"max_num:{max_num_locations},min_num:{min_num_locations},max_gain:{max_gain:.10f},min_gain:{min_gain: .10f}")
    
    for siot in siots:
        coverage = (len(set(siot.locations) & set(request_locations))-min_num_locations) / (max_num_locations - min_num_locations)
        gain = (abs(siot.gain) - min_gain) / (max_gain - min_gain)
        # print (f"Coverage{coverage}, gain: {gain}")
        ccm= coverage * gain
        if max_ccm < ccm:
            max_ccm = ccm
            max_index = siot.index
    
    #print (f"SIoT{max_index} for the first SIoT")
    S.add(siots[max_index])
    return S, max_num_locations


def compute_spccm(siots, S, request_locations, max_num_locations):
    """
    Compute the Spatial Phase-Coverage Contribution Metric (SPCCM) for each SIoT.
    
    Args:
        siots: List of SIoT objects.
        selected_siot: The SIoT selected from CCM.
        S: Set of selected SIoTs.
        request_locations: Set of requested monitoring locations.

    Returns:
        best_siot: The SIoT with the highest SPCCM.
    """
    wavelength=0.125

    covered_locations = set(loc for siot in S for loc in siot.locations)
    #print(f"First covered location:{covered_locations}")

    #SPCCM to select SIoTs into S
    
    #Phase of each SIoT
    phase_terms = np.array([np.exp(1j*(2 * np.pi * siot.distance * np.cos(np.radians(siot.angle))) / wavelength) for siot in siots])
    #phase_terms = np.array([np.exp(1j * siot.angle) for siot in siots])
    
    #print (f"Phase{phase_terms}")
    while not set(covered_locations).issuperset(set(request_locations)):
        max_spccm = 0
        max_index = None
        for siot in siots:
            if siot.index not in {s.index for s in S}:
                phase_alignment = np.abs(phase_terms[siot.index] + sum(phase_terms[[siot.index for siot in S]])) / (len(S) + 1)
                additional_coverage = len((set(siot.locations) & set(request_locations)) - covered_locations) / max_num_locations
                spccm=phase_alignment* additional_coverage
                #print(f"SIoT {siot.index} locations: {set(siot.locations)}")
                #print(f"Intersection: {covered_locations & (set(siot.locations))}")
                #print(f"Union: {covered_locations | (set(siot.locations))}")
                #print(f"SIoT {siot.index}, phase:{phase_alignment}, add_coverage:{additional_coverage}, spccm:{spccm}")
                if max_spccm < spccm:
                    max_spccm = spccm
                    max_index = siot.index
        #print(f"SIoT{max_index} add into set S")
        S.add(siots[max_index])
        covered_locations.update(siots[max_index].locations)
        #print(f"covered locations update:{covered_locations}")
        
    
    print("All request locations are now covered! End Phase 1-------------------------------")
    return S



if __name__ == '__main__':
    main()