#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Werewolfzy

import sys
import getopt
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import os



def usage():
    print(
"""
usage: python [{0}] ... [-m model_file | -d data_file | -o out_file]  ...
参数说明:
-m     : 模型文件，Model file
-d     : 待预测个体数据文件，Individual data file to be predicted
-o     : 数据保存名字，The name of the result file
-h     : 帮助信息
""".format(sys.argv[0]))

def main():

    opts,args = getopt.getopt(sys.argv[1:],"hm:d:o:")
    model_file = ""
    data_file = ""
    out_file = "result.txt"


    for op,value in opts:
        if op == '-m':
            model_file = value
        elif op == "-d":
            data_file = value
        elif op == "-o":
            out_file = value
        else:
            usage()
            sys.exit()
            return


    # print(genotype_file,phenotype_file,significant_snp,freq_cutoff,parent_info,out_file)


    # print(df)



    loaded_model = joblib.load(model_file)
    path = data_file  # 数据文件路径
    data = pd.read_csv(path, encoding="gbk", sep='\t')

    file_name = os.path.basename(model_file)

    if file_name == 'model_xgboost_sheep.dat':
        label_dict = {
            1: 'AW',
            2: 'DM',
            3: 'GB',
            4: 'GG',
            5: 'GJ',
            6: 'GOL',
            7: 'HB',
            8: 'HZ',
            9: 'JC',
            10: 'JL',
            11: 'JZ',
            12: 'LKZ',
            13: 'NL',
            14: 'QK',
            15: 'QL',
            16: 'QOL',
            17: 'SG',
            18: 'ZG',
            19: 'ZK',
            20: 'ZX'
        }

        pass
    elif file_name == 'model_xgboost_pig.dat':
        label_dict = {
            1: 'DN',
            2: 'DQ-W',
            3: 'ML-W',
            4: 'SB',
            5: 'TP-DQ',
            6: 'TP-GN',
            7: 'TP-GZ',
            8: 'TP-LZ'
        }

        pass
    elif file_name == 'model_xgboost_chicken.dat':
        label_dict = {
            1: "AB1",
            2: "BCY1",
            3: "CK1",
            4: "HT1",
            5: "LN1",
            6: "T-BY1",
            7: "T-CT1",
            8: "T-GB1",
            9: "T-NX1",
            10: "T-RKZ1",
            11: "T-SC1",
            12: "T-SN1",
            13: "WLSB1",
            14: "WXB1"
        }

        pass
    else:
        pass








    data.rename(columns={'#CHROM': 'CHROM'}, inplace=1)
    data1 = data.copy()
    data2 = data1.drop(labels=['CHROM', 'POS', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT'], axis=1)
    data3 = data2.set_index('ID')
    data4 = data3.T
    data5 = data4.replace({"0/0": 1, "0/1": 2, "1/1": 3, "1月1日": 3})
    data7 = data5.replace("./.", np.NAN)
    data8 = xgb.DMatrix(data7)
    ypred = loaded_model.predict(data8)


    ypred_labels = [label_dict[label] for label in ypred]

    data9 = pd.DataFrame(ypred_labels, columns=['Predicted_Label'])

    x_values = data7.index.tolist()
    data10 = pd.DataFrame({'x_values': x_values})

    data_result = pd.concat([data10, data9], axis=1)
    data_result = data_result.rename(columns={'x_values': 'ID'})
    data_result.to_csv(out_file, sep='\t', index=False)
    print('程序运行结束，请查看结果文件')
    print('The program is finished, please view the result file')








if __name__ == '__main__':
    main()













