# Relevance-based Margin for Contrastively-trained Video Retrieval Models
In this repo, we provide code and pretrained models for the paper ["**Relevance-based Margin for Contrastively-trained Video Retrieval Models**"](https://arxiv.org/abs/2204.13001) which has been accepted for presentation at the ACM International Conference on Multimedia Retrieval ([ICMR 2022](https://www.icmr2022.org/)).
We also provide code and pretrained models for RelevanceMargin-HGR [here](https://github.com/aranciokov/RelevanceMargin-HGR-ICMR22).

#### Python environment
The environment used is based on the [JPoSE environment](https://github.com/mwray/Joint-Part-of-Speech-Embeddings/blob/main/JPOSE_environment.yml). To create a Conda environment from RelMarg_environment.yml, type:
```
conda env create -f RelMarg_environment.yml
conda activate JPoSE
```
Then clone the repository and type
```
export PYTHONPATH=src/
```

#### Data
- Features: 
    - **EPIC-Kitchens-100**: [**video features**](https://drive.google.com/file/d/1zfhXvAwG7ivX7hGM6d9tpplcX_65OxJM/view?usp=sharing) and [**text features**](https://drive.google.com/file/d/1Y2bxqZD4d-fHxY3J2R215HhmnmGFUf3w/view?usp=sharing); both from [JPoSE's repo](https://github.com/mwray/Joint-Part-of-Speech-Embeddings). 
    - **YouCook2**: [**video features**](https://drive.google.com/file/d/1zKsdTWla6ZZeFDl8dJX8eTbk2b8PQ2-X/view?usp=sharing) and [**text features**](https://drive.google.com/file/d/1QenKgwSsNT-aPA3E1tvTsasinf4LwuFx/view?usp=sharing); video from [VALUE benchmark](https://value-benchmark.github.io), text are precomputed using code from JPoSE repo.
- Additional:
    - relational files: [EPIC-Kitchens-100](https://drive.google.com/file/d/19OwYx093iBSPUaYTB9hYiAdRkTneJVC8/view?usp=sharing) and [YouCook2](https://drive.google.com/file/d/1F3BCFy0CaM_q0rF9Ja89xW1WwYHJEC6d/view?usp=sharing)
    - relevancy files: [EPIC-Kitchens-100](https://drive.google.com/file/d/1glmRIRZ9y6iiJ--Qk6CR2hRwajR6N0Gb/view?usp=sharing) and [YouCook2](https://drive.google.com/file/d/1MHQ77-1k62WzIDND9XNY74deAZ5OG3ss/view?usp=sharing)

#### Training
To launch a training (with JPoSE) on EPIC-Kitchens-100:
``python -m train.train_jpose_tripletRelBased``
- to use the *proposed relevance margin*, specify ``--rel-margin --all-noun-classes``
- add ``--rgb`` to use only RGB features, ``--rgb-flow`` to use RGB+Flow, otherwise do not add anything to use RGB+Flow+Audio (TBN features)
- to only use cross-modality loss, specify ``--tt-weight 0 --vv-weight 0``
- to only use action-level embedding space, specify ``--noun-weight 0 --verb-weight 0``
- to use a GPU, specify ``--gpu True``
- more options in ``src/parsing/__init__.py`` and the ``src/train/train_{mmen,jpose}_tripletRelBased.py`` files

To train on YouCook2, specify ``--dataset youcook2``. Similar options are available for MMEN baseline.

#### Evaluating
To test a specific checkpoint:
``python -m train.test_jpose_triplet checkpoint``
- use the same options used during training (e.g. if training was performed with RGB-only features, specify ``--rgb``)

#### Pretrained models
*On EPIC-Kitchens-100:*
- MMEN
  - [Baseline reproduced (~48.5 nDCG, ~38.5 mAP)](https://drive.google.com/file/d/17cRJarPwTujdKjrvdfDdbivu8kh3tfl4/view?usp=sharing), [With Relevance Margin (49.6 nDCG, 39.2 mAP)](https://drive.google.com/file/d/1a2O_bsfrHO1cM_fMXhn0LS2ESoWWUMO1/view?usp=sharing)
- JPoSE
  - [Baseline reproduced (~53.5 nDCG, ~44.0 mAP)](https://drive.google.com/file/d/129b4LfjL_P2XrKnjkapmAL0jS3tzF411/view?usp=sharing), [**With Relevance Margin (56.2 nDCG, 45.8 mAP)**](https://drive.google.com/file/d/1vHGJYsoswL-dx-S78foKAvDrFqa5acEF/view?usp=sharing)
  - [Only cross-modality loss, action-level (53.1 nDCG, 43.4 mAP)](https://drive.google.com/file/d/1LiRy-dTQp90OUUdvlNXUtHjNByhDl5LC/view?usp=sharing), [With Relevance Margin (54.7 nDCG, 44.5 mAP)]()
  - [Only cross-modality loss, both action- and PoS-level (53.4 nDCG, 43.7 mAP)](https://drive.google.com/file/d/1Z812z8N0BWLQrvWZeGfgoZFEFOWpb7if/view?usp=sharing), [With Relevance Margin (56.2 nDCG, 45.6 mAP)](https://drive.google.com/file/d/1wZp1trXv0467sq4_VvRvBGX2F8Ja3u7d/view?usp=sharing)
  - [Only RGB features (36.8 nDCG, 28.8 mAP)](https://drive.google.com/file/d/15_lLU71_4CYujsmDbNJurd6JT3chwMdh/view?usp=sharing), [With Relevance Margin (38.4 nDCG, 30.4 mAP)](https://drive.google.com/file/d/1jW2hcqLr0U8oY_gIZX8ZwEXt7-wWCwZM/view?usp=sharing)
  - [RGB+Flow features (49.6 nDCG, 41.0 mAP)](https://drive.google.com/file/d/1k1lf9ZkIzDTZJ3HwlmGE_RqsjKaU_lWm/view?usp=sharing), [With Relevance Margin (52.5 nDCG, 42.8 mAP)](https://drive.google.com/file/d/1hfYyNBM0b0trkvNRh9pLgLK1aG-3pMa2/view?usp=sharing)
  
*On YouCook2:*
- MMEN
  - [Baseline reproduced (~48.5 nDCG, ~38.5 mAP)](https://drive.google.com/file/d/1I35M4ucexWPZ_hin-ErYwvEEW5iANgby/view?usp=sharing), [With Relevance Margin (49.6 nDCG, 39.2 mAP)](https://drive.google.com/file/d/1Kcu4Sh8GGaZTa9JVJXe9if16ckEDockv/view?usp=sharing)
- JPoSE
  - [Baseline reproduced (~53.5 nDCG, ~44.0 mAP)](https://drive.google.com/file/d/1dW_6ntBV9mVdPaEmwpZGDgbXFu1NyPs-/view?usp=sharing), [With Relevance Margin (56.2 nDCG, 45.8 mAP)](https://drive.google.com/file/d/1b65THUJDeZiOKp24drSkUSF3xfHZgiQP/view?usp=sharing)

#### Acknowledgements
We thank the authors of 
 [Chen et al. (CVPR, 2020)](https://arxiv.org/abs/2003.00392) ([github](https://github.com/cshizhe/hgr_v2t)),
 [Wray et al. (ICCV, 2019)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wray_Fine-Grained_Action_Retrieval_Through_Multiple_Parts-of-Speech_Embeddings_ICCV_2019_paper.pdf) ([github](https://github.com/mwray/Joint-Part-of-Speech-Embeddings)),
 [Wray et al. (CVPR, 2021)](https://arxiv.org/abs/2103.10095) ([github](https://github.com/mwray/Semantic-Video-Retrieval))
 for the release of their codebases. We thank [Damen et al. (IJCV, 2021)](https://arxiv.org/abs/2006.13256) and [Li et al. (NeurIPS Track on Datasets and Benchmarks, 2021)](https://arxiv.org/abs/2106.04632) for the release of the EPIC-Kitchens-100 and the YouCook2 features.

## Citations
If you use this code as part of any published research, we'd really appreciate it if you could cite the following paper:
```text
@article{falcon2022relevance,
  title={Relevance-based Margin for Contrastively-trained Video Retrieval Models},
  author={Falcon, Alex and Sudhakaran, Swathikiran and Serra, Giuseppe and Escalera, Sergio and Lanz, Oswald},
  journal={ICMR},
  year={2022}
}
```

## License

MIT License
