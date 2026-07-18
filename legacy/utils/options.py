#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():

    #這是 argparse 模組裡面定義的一個類別（Class）。加上括號 () 就是在呼叫它的建構子（Constructor），從而在記憶體中建立出一個物件實體（Instance）
    parser = argparse.ArgumentParser()
    # hierarchical arguments
    parser.add_argument('--num_cells', type=int, default=9,help="number of cells")
    parser.add_argument('--num_edge_steps', type=int, default=5, help="rounds of edge every per global aggregation")
    # federated arguments
    parser.add_argument('--epochs', type=int, default=50, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=90, help="number of users: K") #[30, 60, 90, 150, 180], 10 per cell
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=128, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=256, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0, help="SGD momentum (default: 0)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--model_size', type=int, default='300', help='model size')
    parser.add_argument('--model_size1', type=int, default='120', help='model size 1')
    parser.add_argument('--model_size2', type=int, default='84', help='model size 2')

    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")
   
    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--cell_radius', type=int, default=300, help='Coverage of each edge server')
    parser.add_argument('--num_antennas', type=int, default=4, help='Number of server-side antennas')
    #parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--num_workers', type=int, default=12, help='number of workers')
    parser.add_argument('--num_runs', type=int, default=200, help='number of runs')
    parser.add_argument('--sim', action='store_true', help='simulation mode')
    parser.add_argument('--debug', action='store_true', default=False, help='debug mode')
    parser.add_argument('--HFL_algo', type=str, default='MDSGD', help='HFL algorithm')
    parser.add_argument('--specific_algo', type=str, default=None, 
                      help='Run only specific algorithm (MDSGD, SGC+CADD, CDC+CADD, SGC+WRDD, CDC+WRDD)')
    # thresholds
    parser.add_argument('--data_class_threshold', type=float, default=0.05, help='data class constraint')
    parser.add_argument('--data_quantity_threshold', type=int, default=10000, help='minimum data quantity')
    parser.add_argument('--data_balance_threshold', type=int, default=3000, help='maximum data quantity difference')
    parser.add_argument('--max_drop_rate', type=float, default=0.4, help='maximum dropout rate')
    parser.add_argument('--time_threshold', type=float, default=10, help='time threshold') #range (2~10)
    parser.add_argument('--dropout_penalty', type=float, default=40, help='dropout penalty')

    # communication parameters
    parser.add_argument('--total_power_budget', type=int, default=50, help='total power budget')
    parser.add_argument('--power_ratio', type=float, default=0.4, help='common part power ratio')
    parser.add_argument('--noise_power', type=float, default=8.004e-14, help='noise power (k*T*B where k=1.38e-23, T=290K, B=20MHz)')
    parser.add_argument('--quantization_bit', type=int, default=32, help='quantization bit')




     # 在 args_parser 中加入
    parser.add_argument('--server_gpu_mem', type=int, default=8192, help="GPU memory per server in MB")

    # PPTX formulation parameters
    parser.add_argument('--cost_bw', type=float, default=1.0, help='unit cost for bandwidth usage')
    parser.add_argument('--cost_act', type=float, default=1.0, help='unit cost for expert activation')
    parser.add_argument('--cost_fwd', type=float, default=1.0, help='unit cost for data forwarding')
    parser.add_argument('--default_wired_rate', type=float, default=1e9, help='default wired rate between MEC servers')
    parser.add_argument('--uplink_time_budget', type=float, default=None, help='uplink time budget in seconds; if omitted, derive from task deadlines')
    parser.add_argument('--bandwidth_time_fraction', type=float, default=0.2, help='fraction of the tightest task deadline used to derive bandwidth')
    parser.add_argument('--min_bandwidth', type=float, default=0.0, help='minimum derived RSMA group bandwidth')
    parser.add_argument('--max_group_size', type=int, default=4, help='maximum number of IoTs in each RSMA group')
    parser.add_argument('--min_rate', type=float, default=1.0, help='minimum common/private transmission rate')
    parser.add_argument('--output_token_bits', type=float, default=16.0, help='bits per output token')
    parser.add_argument('--default_feature_bits', type=float, default=8000.0, help='default feature volume in bits')
    parser.add_argument('--loss_threshold', type=float, default=1.0, help='default performance loss threshold')

    # External input files for formulation runner
    parser.add_argument('--tasks_file', type=str, default=None, help='JSON file containing task graphs/subtasks')
    parser.add_argument('--servers_file', type=str, default=None, help='JSON file containing MEC server specs')
    parser.add_argument('--experts_file', type=str, default=None, help='JSON file containing expert specs')
    parser.add_argument('--devices_file', type=str, default=None, help='JSON file containing IoT/device snapshots')
    args = parser.parse_args()
    return args
