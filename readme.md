The repository is for the paper "Uncertainty Calibration for Counterfactual Propensity Estimation in Recommendation".


**Summary of the paper:**
---
The paper first identifies that propensity scores for recommendations are often miscalibrated and proposes using calibration as a quality metric for evaluating these scores. Through the implementation of propensity score calibration, we observed significant improvements in recommendation results.

**How to use the code in practice:**
---
First install the environment using the shell command:
```shell
conda env create -f environment.yml
```
Then you need to first train a propensity estimation model:
```python
python train_propensity.py --dataset coat
```
and you will find a ckpt file int eh propensity/saved_model/ directory. And you use the model to generate the propensity scores using gen_propensity.py.
```python
python gen_propensity.py --dataset coat 
```
For platt-scaling method, you need to use python platt_scale.py to calibrate the propensity scores.

After you get the propensity scores, you can train a recomendation model using IPS, DR, MRDR method. e.g.
```python
python train_rec_with_ips.py --dir Platt_Scaling --dataset coat --path propensity/Platt_Scaling/coat.pt
