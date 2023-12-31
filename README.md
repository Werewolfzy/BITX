# BITX (Breed identification model of Tibetan livestock based on xgboost)
`BITX` is a powerful command line tool designed to swiftly and precisely identify various species of Tibetan livestock in the vast Tibetan Plateau region of China.



## Getting Started
0.Git clone BITX

In order to download `BITX`, you should clone this repository via the commands:
```
git clone https://github.com/Werewolfzy/BITX.git
cd BITX
```

1.Prepare the environment：

This script depends on the Python3 software and requires the following dependency packages:
```
pip install -r requirements.txt
```

2.Prepare the data file.

The data file consists of highly informative SNPs genomic information generated by the software, which is essential for the accurate identification of all individuals.

To access the SNPs data, navigate to the SNPs folder located within each species folder.  As an example, the SNPs file for Tibetan sheep can be found at /Tibetan_sheep/SNPs/1192SNPs_ovis3.txt.

For a better understanding of the file format, you can refer to the test folder, specifically the BlindSG6.txt file.



3.The command line
```
python BITX.py -m [model_file] -d [data_file] -o [out_file  (The default is result.txt)]
```
For example, in linux
```
python ./BITX.py -m ./Tibetan_sheep/model/model_xgboost_sheep.dat -d ./Tibetan_sheep/Blind_test_data/BlindSG6.txt -o result.txt
```
in windows
```
python .\BITX.py -m .\Tibetan_sheep\model\model_xgboost_sheep.dat -d .\Tibetan_sheep\Blind_test_data\BlindSG6.txt -o result.txt
```










