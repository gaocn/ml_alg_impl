# encoding: utf-8
"""
 Author: govind
 Date: 2018/4/8
 Description: 
"""

import random
from  multiprocessing.pool import Pool

"""
   sys_topic            line_count
log_mbank.json              4
log_qpay.json               1
ebflog_qpay.json            2
log_perbank.json            5
log_perbankapps.json        3
"""

# line_count=0
# with open('ebflog_qpay.json', encoding='utf-8') as fd:
#     for line in fd:
#         line_count += 1
#         print(line_count)

samples = [
    {"log_mbank.json": 4},
    {"log_qpay.json": 1},
    {"ebflog_qpay.json": 2},
    {"log_perbank.json": 5},
    {"log_perbankapps.json": 3},
]


def do_generate_logs(sample):
    if not sample:
        print('No Input Parameter!')
        return
    sample_file_name, sample_count = None, None
    for sample_file_name, sample_count in sample.items():
        print('Process to generate {0}, sample count {1}'.format(sample_file_name, sample_count))

    samples = []
    # read sampel json into memory
    for line in open(sample_file_name, 'r', encoding='utf-8'):
        samples.append(line)
    print("all samples' line numbers:{0}".format(len(samples)))

    # generate sample logs in to respective files
    writer = open('sample_{0}'.format(sample_file_name), 'w', encoding='utf-8')
    count = 1
    while True:
        rand_line_num = random.randint(1, sample_count)
        print('[{0}][Count={1}]random line numer: {2}'.format(sample_file_name, count, rand_line_num))
        writer.write(samples[rand_line_num - 1])
        writer.write('\n')
        count += 1

    writer.flush()
    writer.close()


def generate_logs():
    with Pool(5) as pools:
        pools.map(do_generate_logs, samples)


if __name__ == "__main__":
    generate_logs()