# Efficient Communication-Coordination
This repository contains the code and data for the [CoNLL 2024](https://conll.org/) paper titled "Is Structure Dependence Shaped for Efficient Communication?: A Case Study on Coordination" (Kajikawa et al., 2024; 🏆Best Paper Award).\
[[Paper link](https://aclanthology.org/2024.conll-1.23/)]
[[arXiv](https://arxiv.org/abs/2410.10556)]

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


The data and results we used are available as zip files at [this google drive](https://drive.google.com/drive/folders/12mRPlKXXQKueYbITQYDkeRKlHvyMrS8T?usp=sharing).

## Citation
```
@inproceedings{kajikawa-etal-2024-structure,
    title = "Is Structure Dependence Shaped for Efficient Communication?: A Case Study on Coordination",
    author = "Kajikawa, Kohei  and
      Kubota, Yusuke  and
      Oseki, Yohei",
    editor = "Barak, Libby  and
      Alikhani, Malihe",
    booktitle = "Proceedings of the 28th Conference on Computational Natural Language Learning",
    month = nov,
    year = "2024",
    address = "Miami, FL, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.conll-1.23/",
    doi = "10.18653/v1/2024.conll-1.23",
    pages = "291--302"
}
```
