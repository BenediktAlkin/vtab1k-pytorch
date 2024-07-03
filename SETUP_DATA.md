# Setup VTAB-1K

We use the preprocessed VTAB-1K dataset from [RepAdapter](https://github.com/luogen1996/RepAdapter?tab=readme-ov-file#data-preparation).
It is fully preprocessed and can be found [here](https://drive.google.com/file/d/1yZKwiKdsBzTfBgnStRveYMokc7GMMd5p). 
You can also download it via command line:

```
pip install gdown
gdown 1yZKwiKdsBzTfBgnStRveYMokc7GMMd5p
```

Then unzip it with `unzip vtab-1k.zip` and (optionally) remove the zip `rm vtab-1k.zip`

## Store each dataset as zip

Unzipping all 19 datasets takes quite a while. If you want to copy the dataset before training from some 
"global storage" (big but slow storage; e.g. a shared network file system) to a "local storage" (fast but small; e.g.
a local SSD on a compute node), you can also store each dataset as zip file. It will then be unzipped to the local 
storage before training starts.

You can create dataset-level zips by first unzipping the `vtab-1k.zip` from above, creating individual zips and 
then deleting the unzipped dataset:

```
cd vtab-1k

# create zips
zip -r caltech101.zip caltech101
zip -r cifar.zip cifar
zip -r dtd.zip dtd
zip -r oxford_flowers102.zip oxford_flowers102
zip -r oxford_iiit_pet.zip oxford_iiit_pet
zip -r svhn.zip svhn
zip -r sun397.zip sun397
zip -r patch_camelyon.zip patch_camelyon
zip -r eurosat.zip eurosat
zip -r resisc45.zip resisc45
zip -r diabetic_retinopathy.zip diabetic_retinopathy
zip -r clevr_count.zip clevr_count
zip -r clevr_dist.zip clevr_dist
zip -r dmlab.zip dmlab
zip -r kitti.zip kitti
zip -r dsprites_loc.zip dsprites_loc
zip -r dsprites_ori.zip dsprites_ori
zip -r smallnorb_azi.zip smallnorb_azi
zip -r smallnorb_ele.zip smallnorb_ele

# removed unzipped
rm -rf caltech101
rm -rf cifar
rm -rf dtd
rm -rf oxford_flowers102
rm -rf oxford_iiit_pet
rm -rf svhn
rm -rf sun397
rm -rf patch_camelyon
rm -rf eurosat
rm -rf resisc45
rm -rf diabetic_retinopathy
rm -rf clevr_count
rm -rf clevr_dist
rm -rf dmlab
rm -rf kitti
rm -rf dsprites_loc
rm -rf dsprites_ori
rm -rf smallnorb_azi
rm -rf smallnorb_ele
```
