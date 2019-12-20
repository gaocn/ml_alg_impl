# encoding: utf-8
"""
 Author: govind
 Date: 2018/4/8
 Description: 
"""
from multiprocessing.pool import Pool
import random
from socket import *

#####################################################
HOST = '10.233.87.241'
PORT = 2300
samples = [
    "data/sample_log_mbank.json",
    "data/sample_log_qpay.json",
    "data/sample_ebflog_qpay.json",
    "data/sample_log_perbank.json",
    "data/sample_log_perbankapps.json",
    # read log files more than once!
    "data/sample_log_mbank.json",
    "data/sample_log_qpay.json",
    "data/sample_ebflog_qpay.json",
    "data/sample_log_perbank.json",
    "data/sample_log_perbankapps.json"
]

ADDR = (HOST, PORT)
client = socket(AF_INET, SOCK_DGRAM)
#####################################################


def udp_send(data):
    client.sendto(data.encode(encoding='utf-8'), ADDR)
    print('[Log Sender]: {0}'.format(data))


def pipeline(log_file):
    if not log_file:
        print('log_file is None')
        return
    print('[Pipeline] Start to send logs!')
    for line in open(log_file, mode='r', encoding='utf-8'):
        udp_send(line)


def log_sender(n_process=5):
    if n_process > len(samples):
        print('No More Than {0} Processes'.format(len(samples)))
        return
    logs_to_send = random.sample(samples, n_process)
    print('[Logs_to_ send]', logs_to_send)
    with Pool(processes=n_process) as pools:
        pools.map(pipeline, logs_to_send)


if __name__ == "__main__":
    log_sender(n_process=5)