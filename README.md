# Soft Prompts

## Setup

We assume you are running soft-prompts on Linux with GPU available and anaconda3 installed.
To set up the environment, please run

```shell
conda create -n soft-prompts python=3.7
conda activate soft-prompts
conda install -y pytorch==1.7.0 cudatoolkit=11.0 -c pytorch
pip install transformers==4.0.0 pyyaml tqdm
```

## Data

The `prompts` folder contains the prompts we used for T-REx, Google-RE, and ConceptNet.
The `db` folder contains the relations we used for experiments.
We further proprocessed T-REx to split it into train, dev, and test subsets.

## Experiment

To replicate our results on T-REx extended datasets with BERT-large-cased LM, run the following commands:

```shell
git clone git@github.com:hiaoxui/soft-prompts
cd soft-prompts
python3 -m soft_prompts.run.experiment config.yaml
```

## Paper

You can read our paper on [arXiv](https://arxiv.org/abs/2104.06599).

You can cite our paper by

```bibtex
@inproceedings{qin-eisner-2021-learning,
    title = "Learning How to Ask: Querying {LM}s with Mixtures of Soft Prompts",
    author = "Qin, Guanghui  and
      Eisner, Jason",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.410",
    pages = "5203--5212",
}
```

