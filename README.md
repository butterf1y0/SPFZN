1、This repository contains a subset of the complete dataset used in our study. 
These sample data are provided for reference only. 
To obtain the complete dataset, please contact: 240505011015@zufe.edu.cn.

2、The training data directory must contain all three of the following files:
train.dat.apc
valid.dat.apc
test.dat.apc

3、Prediction inputs must follow the format illustrated below, with aspect terms enclosed by [B-ASP] and [E-ASP]:
"The [B-ASP]pizza[E-ASP] is delicious, but the [B-ASP]staff[E-ASP] is rude.",
"I love the [B-ASP]location[E-ASP], but the [B-ASP]price[E-ASP] is too high."


4、For instructions on installing the core components, details of the PyABSA architecture, 
and additional format requirements for training and prediction data, please refer to:

📄 Project: https://github.com/yangheng95/PyABSA
📚 Documentation: https://pyabsa.readthedocs.io/


5、References：
@inproceedings{YangZL23,
    author = {Heng Yang and Chen Zhang and Ke Li},
    title = {PyABSA: A Modularized Framework for Reproducible Aspect-based Sentiment Analysis},
    booktitle = {Proceedings of the 32nd ACM International Conference on Information and Knowledge Management (CIKM 2023)},
    pages = {5117--5122},
    year = {2023},
    doi = {10.1145/3583780.3614752}
}
