# Prototype-based-Memory-Network
The labels and codes for [Aerial Scene Understanding in The Wild: Multi-Scene Recognition via Prototype-based Memory Networks]()

## Usage
download [MAI_dataset](https://drive.google.com/drive/folders/1xMWXxDeELmGKBdBZopSzk4rTpw7kqwzb?usp=sharing) and unzip ```images.zip```. To learn scene prototypes, [AID](https://captain-whu.github.io/AID/) and [UCM](http://weegee.vision.ucmerced.edu/datasets/landuse.html) datasets are required. The data directory structure should be as follows:
```
  path/to/data/
    mai/
      configs/        # data split for UCM2MAI and AID2MAI
      images/         # images     
      label_list.txt  # indices of scene labels
      multilabel.mat  # scene labels
    AID_dataset/      # AID dataset
      Airport/
      Beach/
      ...
    UCM_dataset/      # UCM dataset
      agricultural/
      airplane/
      ...
```

## Citation
If you find they are useful, please kindly cite the following:
```
@article{hua2021prototype,
  title={Aerial Scene Understanding in The Wild: Multi-Scene Recognition via Prototype-based Memory Networks},
  author={Hua, Yuansheng and Mou, Lichao and Lin, Jianzhe and Heidler, Konrad and Zhu, Xiao Xiang},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  year={in press}
}
```
