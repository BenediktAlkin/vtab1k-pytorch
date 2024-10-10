# MIM-Refiner

To reproduce the results of [MIM-Refiner](https://arxiv.org/abs/2402.10093) run the pipelines from [src/pipelines/mimrefiner](https://github.com/BenediktAlkin/vtab1k-pytorch/tree/main/src/pipelines/mimrefiner)

# Vision-LSTM

To reproduce the results of [Vision-LSTM](https://arxiv.org/abs/2406.04303) run the pipelines from 
- [src/pipelines/vil](https://github.com/BenediktAlkin/vtab1k-pytorch/tree/main/src/pipelines/vil) (Vision-LSTM)
- [src/pipelines/vim](https://github.com/BenediktAlkin/vtab1k-pytorch/tree/main/src/pipelines/vim) (Vision-Mamba)
- [src/pipelines/deit](https://github.com/BenediktAlkin/vtab1k-pytorch/tree/main/src/pipelines/deit) (Vision-Transformer)


# EVA

To reproduce the results of [EVA](https://arxiv.org/abs/2410.07170) run the pipelines from [src/pipelines/peft](https://github.com/BenediktAlkin/vtab1k-pytorch/tree/main/src/pipelines/peft).

To train with EVA, you need to first run the PCA for the data-driven weight initialization [src/pipelines/peft/eva/eva_pca](https://github.com/BenediktAlkin/vtab1k-pytorch/tree/main/src/pipelines/peft/eva/eva_pca).
This will store the outputs of the PCA in the `output_uri` that you defined in the `static_config.yaml`.

Afterwards you can train with  [EVA pipelines](https://github.com/BenediktAlkin/vtab1k-pytorch/tree/main/src/pipelines/peft/eva)
which define from where to load the PCA outputs via the `pca_rel_uri` field of the `trainer`
(e.g. `model=dinov2_vitg14__dataset={dataset}_train800__rank=32_ipcabs=16384.th`). Note that the PCA number of ranks is
independent of the number of ranks that are used in EVA. By default we calculate more PCA ranks (ranks=32) than needed
for EVA training.
