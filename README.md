# XLand-100B: A Large-Scale Multi-Task Dataset for In-Context Reinforcement Learning

Official code for the 'XLand-100B: A Large-Scale Multi-Task Dataset for In-Context Reinforcement Learning' paper. We provide the utilities used to collect the datasets as well as the code used for experiments with the baselines, namely AD and DPT. As these parts are semantically unrelated, they are separated into separate directories for simplicity (in the cleanrl style).

Both XLand-100B and XLand-Trivial-20B hosted on public S3 bucket and freely available for everyone under CC BY-SA 4.0 Licence. See the README in each directory for instructions.

## Downloading the datasets

We advise starting with Trivial dataset for debugging due to smaller size and faster downloading time. Both datasets have an identical structure. For additional details we refer to the paper. 

Datasets can be downloaded with the curl utility (or any other like wget) as follows:
```commandline
# XLand-Trivial-20B, approx 60GB size
curl -L -o xland-trivial-20b.hdf5 https://sc.link/A4rEW

# XLand-100B, approx 325GB size
curl -L -o xland-100b.hdf5 https://sc.link/MoCvZ
```