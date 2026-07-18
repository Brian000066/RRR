config = {
    'num_devices' : 100, #experimental variables
    'BANDWIDTH': 1 * 10^6,  #MHZ
    'num_antennas': 4, 
    'data_quantity_constraint': 12000, #(change in main)
    'data_coverage_constraint': 0.6,
    'label_class': 10, #(change in main)
    'theta_SDCI': 0.4,
    'count_limit': 4,
    'com_weight': 1, 
    'cmp_weight': 1e-9, #使用GIGA Flops計算
    'importance_epsilon': 0.6,
    'delay_constraint': 1, #experimental variables
    'pvt_ratio': 0.4,
    'cmn_ratio': 0.6
}
#math
PI = 3.14159

# powerBS = 6.31 #Watt
powerBS = 15.62
AWGNoise = pow(10,-17.4) # dBm/Hz Noise power spectral density, ThermalNoise

#configure setting
minSinr = -9.478
maxDataRate = 1500

#MCS to data rate (kbps)
QPSK_1 = 0.5 #19.90
QPSK_2 = 1.2 #30.48
QPSK_3 = 2.4 #49.02
QPSK_4 = 3.5 #78.22
QPSK_5 = 4.2 #114.03
QPSK_6 = 5.1 #152.89
QAM16_7 = 6 #192.00
QAM16_8 = 7.6 #248.89
QAM16_9 = 8.8 #312.89
QAM16_10 = 9.5 #355.05
QAM16_11 = 10.2 #432.00
QAM16_12 = 20.4 #507.43
QAM16_13 = 30.3 #588.19
QAM16_14 = 40 #667.43
QAM16_15 = 50 #722.29


def rate_mapping(min_sinr): #SINR to rate
    if min_sinr <= -9.478 :
        rate = QPSK_1
    elif min_sinr > -9.478 and min_sinr <= -6.658:
        rate = QPSK_2
    elif min_sinr >-6-658  and min_sinr <= -4.098:
        rate = QPSK_3
    elif min_sinr > -4.098 and min_sinr <= -1.798:
        rate = QPSK_4
    elif min_sinr > -1.798 and min_sinr <= 0.399:
        rate = QPSK_5
    elif min_sinr >  0.399 and min_sinr <= 2.424:
        rate = QPSK_6
    elif min_sinr >  2.424 and min_sinr <= 4.489:
        rate = QAM16_7
    elif min_sinr >  4.489 and min_sinr<= 6.367:
        rate = QAM16_8
    elif min_sinr >  6.367 and min_sinr <= 8.456:
        rate = QAM16_9
    elif min_sinr >  8.456 and min_sinr <= 10.266:
        rate = QAM16_10
    elif min_sinr >  10.266 and min_sinr <= 12.218:
        rate = QAM16_11
    elif min_sinr >  12.218 and min_sinr <= 14.122:
        rate = QAM16_12
    elif min_sinr >  14.122 and min_sinr <= 15.849:
        rate = QAM16_13
    elif min_sinr >  15.849 and min_sinr <= 17.786:
        rate = QAM16_14
    else:
        rate = QAM16_15
    return rate