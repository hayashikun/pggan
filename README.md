# PGGAN

```
$ ssh-keygen -t rsa
$ git clone git@github.com:hayashikun/pggan.git
$ cd pggan
$ pip install -U pip & pip install -r requirements.txt
$ aws configure
$ export PYTHONPATH="/home/ubuntu/pggan:$PYTHONPATH"
$ python pggan/main.py load_dataset
$ nohup python pggan/main.py train &

$ watch -n 1 nvidia-smi
```