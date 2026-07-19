import random
import numpy as np
from input_settings import get_siot
from input_settings import SIoT
from phase1 import phase1
from config import rate_mapping
from itertools import combinations
from phase2 import phase2, check_distributed_beamforming_constraint, compute_collaborative_cost
from phase3 import phase3
from input_settings import get_siot , parse_siots_from_json,parse_requests_from_json


def main():
    num_batches = 1000
    total_cost = 0
    total_bandwidth_cost=0
    total_computation_cost=0
    total_communication_cost=0
    total_common_message_data_size=0
    total_collaborative_cost=0
    total_collaborative_coverage=0
    total_RSMA_coverage=0
    total_beamforming_gain=0
    skip_batch=0
    total_local_computation_cost=0
    total_selected_siot=0

    for batch_id in range(num_batches):
        print(f"\n=== Processing Batch {batch_id+1}/{num_batches} ===")

        siots= parse_siots_from_json("siots.json",batch_id)
        requests=parse_requests_from_json("requests.json",batch_id)
        # siot_set,unselected_siot_set =phase1(siots, request_locations)
        # rsma_groups, collaborative_groups,unselected_siot_set=phase2(siot_set,unselected_siot_set)
        # phase3(rsma_groups, collaborative_groups,unselected_siot_set)

        # 累加所有 requests 的總成本
        batch_total_cost = 0
        batch_total_bandwidth_cost=0
        batch_total_computation_cost=0
        batch_total_communication_cost=0
        batch_total_common_message_data_size=0
        batch_total_collaborative_cost=0
        batch_total_collaborative_coverage=0
        batch_total_RSMA_coverage=0
        batch_total_beamforming_gain=0
        skip_request=0
        batch_total_local_computation_cost=0
        batch_total_selected_siot=0

        # 遍歷每個 request
        for request_id, request_locations in requests.items():
            print(f"\n=== Processing Request {request_id} ===")

            # Phase 1: 選擇 SIoTs
            siot_set, unselected_siot_set = phase1(siots, request_locations)

            # Phase 2: 分組 SIoTs
            rsma_groups, collaborative_groups, unselected_siot_set = phase2(siot_set, unselected_siot_set)

            # Phase 3: 計算總成本
            request_cost, bandwidth_cost,computation_cost,communication_cost,RSMA_coverage,common_message_data_size,collaborative_cost,collaborative_coverage,beamforming_gain,local_computation_cost,selected_siot = phase3(rsma_groups, collaborative_groups, unselected_siot_set)
            batch_total_cost += request_cost  # 累加成本
            batch_total_bandwidth_cost+=bandwidth_cost
            batch_total_computation_cost+=computation_cost
            batch_total_communication_cost+=communication_cost
            batch_total_common_message_data_size+=common_message_data_size
            batch_total_RSMA_coverage+=RSMA_coverage
            batch_total_collaborative_coverage+=collaborative_coverage
            batch_total_collaborative_cost+=collaborative_cost
            if beamforming_gain ==0:
                skip_request+=1
            batch_total_beamforming_gain+=beamforming_gain
            batch_total_local_computation_cost+=local_computation_cost
            batch_total_selected_siot+=selected_siot
            print(f"Request {request_id} Cost: {request_cost:.4f}")
    
        # 累加所有 batch 的總成本
        total_cost += batch_total_cost
        total_bandwidth_cost += batch_total_bandwidth_cost
        total_computation_cost += batch_total_computation_cost
        total_communication_cost += batch_total_communication_cost
        total_common_message_data_size+=batch_total_common_message_data_size
        total_collaborative_cost+=batch_total_collaborative_cost
        total_collaborative_coverage+=batch_total_collaborative_coverage
        total_RSMA_coverage+=batch_total_RSMA_coverage
        if len(requests)!=skip_request:
            total_beamforming_gain+=(batch_total_beamforming_gain / len(requests)-skip_request)
        else:
            skip_batch+=1
        total_local_computation_cost+=batch_total_local_computation_cost
        total_selected_siot+=batch_total_selected_siot
    # 計算平均成本
    avg_total_cost = total_cost / num_batches
    avg_bandwidth_cost = total_bandwidth_cost / num_batches
    avg_computation_cost = total_computation_cost / num_batches
    avg_communication_cost = total_communication_cost / num_batches
    avg_common_message_data_size=total_common_message_data_size / num_batches
    avg_collaborative_coverage=total_collaborative_coverage / num_batches
    avg_RSMA_coverage=total_RSMA_coverage / num_batches
    avg_collaborative_cost=total_collaborative_cost / num_batches
    avg_local_computation_cost=total_local_computation_cost / num_batches
    avg_selected_siot=total_selected_siot / num_batches

    if num_batches != skip_batch:
        avg_beamforming_gain=total_beamforming_gain / (num_batches-skip_batch)
    else:
        avg_beamforming_gain=0

    print("\n=== Average Cost Over All Batches ===")
    print(f"Average Total Cost: {avg_total_cost:.4f}")
    print(f"Average Bandwidth Cost: {avg_bandwidth_cost:.4f}")
    print(f"Average Computation Cost: {avg_computation_cost:.4f}")
    print(f"Average Communication Cost: {avg_communication_cost:.4f}")
    print(f"Average Common Data Size: {avg_common_message_data_size:.4f}")
    print(f"Average RSMA Coverage: {avg_RSMA_coverage:.4f}")
    print(f"Average Collaborative Cost: {avg_collaborative_cost:.4f}")
    print(f"Average Collaborative Coverage: {avg_collaborative_coverage:.4f}")
    print(f"Average Beamforming Gain: {avg_beamforming_gain:.4f}")
    print(f"Average Local Computation Cost: {avg_local_computation_cost:.4f}")
    print(f"Average Selected SIoT: {avg_selected_siot:.4f}")











if __name__ == '__main__':
    main()