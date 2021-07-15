# Information Obfuscation of Graph Neural Networks

Peiyuan Liao*, Han Zhao*, Keyulu Xu*, Tommi Jaakkola, Geoffrey Gordon, Stefanie Jegelka,
Ruslan Salakhutdinov. ICML 2021.

\* Denotes equal contribution

This repository contains a PyTorch implementation of **G**raph **A**dversaria**L** Networks (GAL).

## Dependencies

 - Compatible with PyTorch 1.7.0 and Python 3.x
 - torch_geometric == 1.6.3 with newest packages specified below:

```
export CUDA=cu92/cu100/cu101/cpu
$ pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+${CUDA}.html
$ pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+${CUDA}.html
$ pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+${CUDA}.html
$ pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+${CUDA}.html
$ pip install torch-geometric
```

## Dataset

- We use FB15k-237 and WN18RR dataset for knowledge graph link prediction. 
- FB15k-237 and WN18RR are included in the `src/Freebase_Wordnet/data` directory. For `POS_tag` and `sense` attribute for WN18RR dataset, we took labels from [Bordes (2013)](https://www.hds.utc.fr/everest/doku.php?id=en:smemlj12), and for FB15k-237, we used entity-level tags from [Moon (2017)](https://github.com/cmoon2/knowledge_graph). Compressed data in `data_compressed` can be found in [repository of CompGCN](https://github.com/malllabiisc/CompGCN).
- We use Movielens-1M dataset for recommendation system link prediction task. You may access the data at [this link.](https://grouplens.org/datasets/movielens/1m/)

## Running

 - FB15k-237/WN18RR:
   - run `preprocess.sh` to unzip data
   - run `run.py -h` for arguments
   - re-run `run.py` with supplied arguments
   - results are reported in log

 - Movielens-1M:
   - create config file under config folder
   - run `exec.py --config_path=config`
   - results are reported in log

 - QM9/Planetoid
   - Run corresponding files under the `benchmarks` dataset

### Reproducing Results

 - FB15k-237/WN18RR:
   - Find `gen_sh.ipynb` under `config` folder
   - Execute the cells and replace path with appropriate path
   - Sequentially execute each generated shell script to obtain results under `log`

 - Movielens-1M:
   - Find `gen_json.ipynb` files under `config` folder
   - Execute the cells and replace path with appropriate path
   - Sequentially execute each generated json script to obtain results under `log`

 - QM9/Planetoid/Cora Visualization
   - Run corresponding files under the `benchmarks` dataset
   - For Cora Visualization, run `Cora_visualization.ipynb` under an interactive environment, and run all cells to obtain the desired results. (tweaking $$\lambda$$ values and the TSNE perplexity parameter will give different results)
   - Parameters are default values for both `planetoid_gal.py` and `qm9_gal.py`

## Our Algorithm and Model
The following figure gives a high-level illustration of our model, **G**raph **A**dversaria**L** Networks (GAL). GAL defends node and neighborhood inference attacks via a min-max game between the task decoder (blue) and a simulated worst-case attacker (yellow) on both the embedding (descent) and the attributes (ascent). Malicious attackers will have difficulties extracting sensitive attributes at inference time from GNN embeddings trained with our framework.
![figure](model.png)

## Visualization of Learned Representations against Attacks
GAL effectively protects sensitive information. Both panels show t-SNE plots of the learned feature representations of a graph under different defense strengths. Node colors represent node classes of the sensitive attribute. The left panel corresponds to the learned representations with no-defense, while the right panel shows the representations learned by GAL. Note that without defense from GAL, the representations on the left panel exhibits a cluster structure of the sensitive attribute, make it easier for potential malicious attackers to infer. As a comparison, with GAL defense, nodes with different sensitive values are well mixed, making it hard for attackers to infer. 
![figure](cora.png)

## Results

![figure](figure.png)

## Citation

If you find the work useful in your research, please consider citing:

```
@InProceedings{pmlr-v139-liao21a,
  title = 	 {Information Obfuscation of Graph Neural Networks},
  author =       {Liao, Peiyuan and Zhao, Han and Xu, Keyulu and Jaakkola, Tommi and Gordon, Geoffrey J. and Jegelka, Stefanie and Salakhutdinov, Ruslan},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {6600--6610},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v139/liao21a/liao21a.pdf},
  url = 	 {http://proceedings.mlr.press/v139/liao21a.html},
  abstract = 	 {While the advent of Graph Neural Networks (GNNs) has greatly improved node and graph representation learning in many applications, the neighborhood aggregation scheme exposes additional vulnerabilities to adversaries seeking to extract node-level information about sensitive attributes. In this paper, we study the problem of protecting sensitive attributes by information obfuscation when learning with graph structured data. We propose a framework to locally filter out pre-determined sensitive attributes via adversarial training with the total variation and the Wasserstein distance. Our method creates a strong defense against inference attacks, while only suffering small loss in task performance. Theoretically, we analyze the effectiveness of our framework against a worst-case adversary, and characterize an inherent trade-off between maximizing predictive accuracy and minimizing information leakage. Experiments across multiple datasets from recommender systems, knowledge graphs and quantum chemistry demonstrate that the proposed approach provides a robust defense across various graph structures and tasks, while producing competitive GNN encoders for downstream tasks.}
}

```
