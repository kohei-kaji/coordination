# Efficient Communication-Coordination
This repository contains the code and data for the [CoNLL 2024](https://conll.org/) paper titled "Is Structure Dependence Shaped for Efficient Communication?: A Case Study on Coordination" (Kajikawa et al., 2024).\
[arXiv](https://arxiv.org/abs/2410.10556)

For any questions, please contact:\
kohei.kajikawa1223@gmail.com


## Requirements
We used `Python==3.11.2`.

Additionally, download the following repositories:
- [artificial languages](https://github.com/rycolab/artificial-languages) ([White and Cotterell, 2021](https://aclanthology.org/2021.acl-long.38/))
- [rnng-pytorch](https://github.com/aistairc/rnng-pytorch) ([Noji and Oseki, 2021](https://aclanthology.org/2021.findings-acl.380/))
    - Note:\
    You need to modify a few files in the `rnng-pytorch` repository as described in `src/rnng-pytorch_/README.md`.
    The modified versions of these files are in the `src/rnng-pytorch_` directory.
    Please replace the corresponding files in the `rnng-pytorch` repository with these modified files.\
    Ryo Yoshida (p.c.) provided me the code for calculating the logliks of each parse. Thanks Ryo!


The data and results we used are available as zip files under the `data` and `result` directories, respectively.

## Citation
```
@inproceedings{kajikawa-etal-2024-is,
    title = "Is Structure Dependence Shaped for Efficient Communication?: A Case Study on Coordination",
    author = "Kajikawa, Kohei  and
              Kubota, Yusuke  and
              Oseki, Yohei",
    booktitle = "Proceedings of the 29th Conference on Computational Natural Language Learning (CoNLL)",
    month = nov,
    year = "2024",
    address = "Miami",
    publisher = "Association for Computational Linguistics",
    url = "",
    doi = "",
    pages = "",
}
```