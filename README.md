## AGD: Active learning based generative design for discovery of wide band gap materials
This software package implements our developed framework AGD for materials design based on active learning. This is the official Python repository. 

[Machine Learning and Evolution Laboratory](http://mleg.cse.sc.edu)<br />
Department of Computer Science and Engineering <br />
University of South Carolina <br />

How to cite:<br />
Rui Xin, Edirisuriya M. D. Siriwardane, Yuqi Song, Yong Zhao, Steph-Yves Louis, Alireza Nasiri, Jianjun Hu
Active learning based generative design for the discovery of wide bandgap materials.2021

# Table of Contents
* [Introduction](#introduction)
* [Installation](#installation)
* [Dataset](#dataset)
* [Usage](#usage)



<a name="introduction"></a>
# Introduction
The package provides 3 major functions:

- Perform active-learning based sampling in whole design latent space (based on Bayesian Optimization).
- Train and evaluate the performance of a screening model (based on Roost).
- Generate material cadidates' cif files based on element substitution (based on ELMD). 

The following paper describes the details of the our framework:
[Active learning based generative design for discovery of wide band gap materials](https://arxiv.org/pdf/2103.00608.pdf)



![](front-pic.png)
<a name="installation"></a>
## Installation
Install any of the relevant packages if not already installed:
* Bayesian Optimization (tested on 1.2.0)
* tensorflow (tested on 2.2.0)
* GATGNN [documentation](https://github.com/superlouis/GATGNN).
* RooSt [documentation](https://github.com/CompRhys/roost).
* Numpy   (tested on 1.18.5)
* Pandas  (tested on 1.1.0) 
* Scikit-learn (tested on 0.21.3) 
* Pytmatgen (tested on 2020.3.13)

Bayesian Optimization, Pytorch, Numpy, Pandas, Scikit-learn, and Pymatgen
```bash
conda install -c conda-forge bayesian-optimization
pip install numpy
pip install pandas
pip install scikit-learn
pip install pymatgen
```


<a name="dataset"></a>
## Dataset
1. Download the compressed file of our dataset using [this link](https://figshare.com/articles/dataset/bd_AML_whole_init_300_csv/14132270)
2. Unzip its content ( two .csv files' and 5 pre-trained models)
3. Move the csv files in your AML_Roost directory. i.e. such that the datapath now exists.

<a name="usage"></a>
## Usage
#### Generate target property material candidates
Once all the aforementionned requirements are satisfied, one can easily generate target property material candidates by running ALSearch.py in the terminal along with the specification of the appropriate flags. At the bare minimum, using --budget to specify the active learning budget, --init to set number of initial samples and --kappa to control balance between exploration and exploitation.
- Example. start active-learning process given budget and kappa.
```bash
python ALSearch.py --budget 50 --kappa 100 --init 300 --candidate_out_path path/you/prefer
```
The generated materials and their predicted property will be automatically generated under specified folder

#### Training a new screening model
 Upon acquire active-learning augumented data, one can train and evaluate a screening model's performance using Roost package and GAN generated dataset.
 The 5 augumented dataset corresponding to Exp1, Exp2_BS, Exp2_AL, Exp3_BS, Exp3_AL in the paper are in /root_path/roost/roost/examples/prepared_training_data/
 
 The 5 pre-trained models in figshare link are corresponding to Exp1, Exp2_BS, Exp2_AL, Exp3_BS, Exp3_AL.
 
 Under roost/roost/examples, you can train and evluate model performance using hold out dataset:
```bash
python roost-predict.py --data-path /root_path/roost/roost/examples/prepared_training_data/Exp3_AL_1153.csv --train --evaluate --val-size 0.2  --epochs 200 --run-id 311
```

#### Evaluating the performance of a model trained by active-learning-augemented data

Independent test dataset is under folder roost/roost/examples/prepared_training_data/
Under roost/roost/examples
```bash
python roost-predict.py --test-path /root_path/roost/roost/examples/prepared_training_data/bd_test_only.csv --regression --evaluate --run-id 3
```

#### Screening Recovery rate test

To test the recovery rate:
```bash
python screen_recover_rate.py
```

