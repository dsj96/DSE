# DSE
Spatio-Temporal Representation Learning with Social Tie forPersonalized POI Recommendation


The implementation of the PPR model(Personalized POI Recommendation).

Paper ```Spatio-Temporal Representation Learning with Social Tie for Personalized POI Recommendation``` for DSE 2021

# Usage:
## Install dependencies 
```pip install -r requirements.txt```
## Installation
Clone this repo.
```
git clone https://anonymous.4open.science/r/DSE-1BEC
chmod u+x train_PPR.sh
```
## Function
1. ```gen_graph.py``` file is used for heterogeneous graph construction. Parameter ```theta``` is $\theta$ in Equ.2, and ```epsilon``` is $\varepsilon$ in Equ.6.
2. ```reconstruct.cpp``` file is used for densifying graph. Parameter ```-threshold``` is $\rho$.
3. ```line.cpp``` file is used for learning latent representations. Parameter ```-size``` is embedding dim $d$.
4. ```train.py``` file is used for training and evaluating the spatio-temporal neural network. Parameter ```DELT_T``` is the time constraint $\tau$, and ```INPUT_SIZE/2``` is the embedding dim $d$. You could also change the ```HIDDEN_SIZE, EPOCH, LR, LAYERS OR TEST_SAMPLE_NUM```. 

# Data
In our experiments, the Foursquare datasets are from https://sites.google.com/site/dbhongzhi/ (update to: https://sites.google.com/view/hongzhi-yin/home). And the Gowalla and Brightkite dataset are from https://snap.stanford.edu/data/loc-gowalla.html and http://snap.stanford.edu/data/loc-Gowalla.html.

## Data Split

In order to make our model satisfactory to the scenario of recommending
for future check-ins, we first sort the check-in records of each user in chronological order. Afterwards, we filter the POIs visited by less than five users and the users with less than ten check-in records. Specifically, for a check-in record ```c=<u,v,t>```, if the user ```u```'s check-in records are less than 5 or the POI ```v``` is checked less than 10 times, this record will be filtered.
Finally we choose the first 80\% of each user’s check-ins in chronological order as train data, the remaining 20\% as test data. The former is stored in ```train_checkin_file.txt``` and the later is stored in ```test_checkin_file.txt``` provided by us.

## Data Format
We utilize the first 80% chronological check-ins of each user as the training set, the remaining 20% as the test data.

train_checkin_file.txt and test_checkin_file.txt :
```<USER ID> \t <CHECKIN TIME> \t <POI ID> \t <LONGITUDE> \t <LATITUDE>```

friendship_file.txt : ```<USER ID>,<USER ID>```

# Training
You can train and evaluate the model by: ```./train_PPR.sh```

Or you can run the specific program file separately, but the parameters should be reasonable (Although we have set some default parameters).
```
eg: 
python3 gen_graph.py --input_path dataset/toyset/ --epsilon 0.5 --theta 24.0
python3 train.py --input_path dataset/toyset/ --input_size 16 --hidden_size 16 --layers 2 --lr 0.001 --delt_t 6.0 --epochs 20 --dr 0.2 --seed 1 --test_sample_num 300

# FOR DSE
python3 main_gcn_ppr.py
```
