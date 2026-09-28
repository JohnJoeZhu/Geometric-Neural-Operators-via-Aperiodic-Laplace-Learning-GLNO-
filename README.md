# Geometric Laplace Neural Operator

prompts.md for instructions.
## Data processing

utils and geometry.py are used for data processing, though not all are useful.
```train
prepare_data.py
```

glno_dataset.py is used **Only** to load dataset, match this to the model data preprocessing.

add new dataset here and add the name to __init__.py
## Model
geometry.py includes geometry function used in model processing

model folder contains different model architectures

If new model is need, add **model here** and add to __init__.py in model folder.

data preprocessing is written into models, you have to match this to the data loader.

GLNO/layers.py is the final version

LNO is without sigma version of GLNO

FNO unimplemented yet.

diffusion_net unimplemented yet.

## Utils

here contains a series of functions for training

If new loss function is need, add **loss function here** .

## training & evaluation
parameters:
```bash
--task=rna,poisson # task type
--distributed # use distributed training TODO: not implemented yet
--local_rank # cuda number if not distributed
--evaluate # evaluate the model on test set after training, must contain load_model
--load_model=logs/poisson/best_model.pth # path to load pre-trained model
```

```bash


conda activate zm_AMG
cd ICLR_GLNO

## vertices

ALL Models Implementation:
Fourier
GINO GEO-FNO LSM
Transformer
GNOT Transolver
Graph
GKNO Sp2GNO

Laplace
GLNO

Other
Unet MLP

DEBUG_MODE=1 #for debug

python train.py --local_rank=1 --task=intra --config=config_segmentation.yaml --dataset_name=data_segmentation.h5

python train.py --task=cortex --local_rank=1 --config=config.yaml

nohup python train.py --task=car --local_rank=3
python train.py --task=car --local_rank=2 --config=config_geofno.yaml
python train.py --task=car --local_rank=2 --config=config_gkno.yaml
python train.py --task=car --local_rank=2 --config=config_gino.yaml
python train.py --task=car --local_rank=1 --config=config_unet.yaml
python train.py --task=car --local_rank=3 --config=config_sp2gno.yaml

python visulize.py --task=car --local_rank=2 --load_model=11_06/GLNO_20_22_29,GEO_FNO_19_32_26,GKNO_10_27_39,LSM_11_41_07,MLP_11_28_21,Unet_10_32_26 --model_list=GLNO,Geo-FNO,GKNO,LSM,MLP,Unet

nohup python train.py --task=rna --local_rank=3
python train.py --task=rna --local_rank=3 --config=config_geofno.yaml

nohup python train.py --task=poisson --local_rank=3
python train.py --task=poisson --local_rank=2 --config=config_geofno.yaml
python train.py --task=poisson --local_rank=2 --config=config_gkno.yaml
python train.py --task=poisson --local_rank=3 --config=config_sp2gno.yaml
python train.py --task=poisson --local_rank=3 --config=config_gino.yaml
python visulize.py --task=poisson --local_rank=1 --load_model=GLNO_18_19_19,GEO_FNO_18_55_30,GKNO_21_31_41,LSM_15_55_49,MLP_18_17_18,Unet_18_17_37 --model_list=GLNO,Geo-FNO,GKNO,LSM,MLP,Unet
python visulize.py --task=poisson --local_rank=1 --load_model=GLNO_18_19_19 --model_list=GLNO

nohup python train.py --task=airfoil --local_rank=2


python train.py --task=cylinder_flow --local_rank=1

python train.py --task=human --local_rank=3

python visulize.py --task=human --local_rank=1 --load_model=GLNO_15_00_16,GEO_FNO_21_14_50,GKNO_20_53_21,GNOT_23_37_51,LSM_21_40_38,MLP_21_07_39,Transolver_23_49_45,Unet_21_16_13 --model_list=GLNO,GeoFNO,GKNO,GNOT,LSM,MLP,Transolver,Unet

python visulize.py --task=human --local_rank=1 --load_model=GLNO_15_00_16,GEO_FNO_21_14_50,GNOT_23_37_51,LSM_21_40_38,Transolver_23_49_45,Unet_21_16_13 --model_list=GLNO,GeoFNO,GNOT,LSM,Transolver,Unet


python visulize.py --task=rna --local_rank=0 --load_model=GLNO_17_27_39,GEO_FNO_19_59_55,GNOT_16_48_56,LSM_16_40_19,GKNO_08_00_00,Transolver_23_54_51,Unet_16_21_19 --model_list=GLNO,GeoFNO,GNOT,LSM,GKNO,Transolver,Unet
## grid

python train.py --task=turbulent --local_rank=1 --config=config_glno.yaml --seed=42

python train.py --task=pendulum/c02 --local_rank=1 --config=config_cno.yaml # fno cno wno lno glno 

python train.py --task=duffing/c0 --local_rank=1 --config=config_fno.yaml
python visulize.py --task=lorenz/rho10 --local_rank=1 --load_model=GLNO1D_21_43_09,CNO1D_21_35_48,FNO1D_21_32_21,WNO1D_21_33_12,LNO1D_21_45_29 --model_list=GLNO,CNO,FNO,WNO,LNO
python visulize.py --task=lorenz/rho10 --local_rank=1 --load_model=GLNO1D_12_00_00,LNO1D_21_45_29,WNO1D_21_33_12,FNO1D_21_32_21,CNO1D_21_35_48 --model_list=GLNO,LNO,WNO,FNO,CNO
python visulize.py --task=lorenz/rho05 --local_rank=1 --load_model=GLNO1D_21_23_21,CNO1D_19_12_14,FNO1D_21_10_10,WNO1D_21_16_37,LNO1D_21_14_05 --model_list=GLNO,CNO,FNO,WNO,LNO
python visulize.py --task=lorenz/rho05 --local_rank=1 --load_model=GLNO1D_21_23_21 --model_list=GLNO
python visulize.py --task=lorenz/rho05 --local_rank=1 --load_model=CNO1D_19_12_14 --model_list=CNO
python visulize.py --task=lorenz/rho10 --local_rank=1 --load_model=01_23/GLNO1D_21_50_12 --model_list=GLNO
python train.py --task=lorenz/rho10 --local_rank=1 --config=config_glno.yaml

python train.py --task=duffing/c05 --local_rank=1 --config=config_glno.yaml
python visulize.py --task=duffing/c05 --local_rank=1 --load_model=GLNO1D_16_35_47,LNO1D_20_50_26,WNO1D_20_48_16,FNO1D_20_47_16,CNO1D_20_59_17 --model_list=GLNO,LNO,WNO,FNO,CNO
python visulize.py --task=duffing/c05 --local_rank=1 --load_model=01_24/GLNO1D_21_57_10 --model_list=GLNO

python visulize.py --task=diffusion --local_rank=1 --load_model=GLNO2D_23_10_30 --model_list=GLNO
python visulize.py --task=diffusion --local_rank=2 --load_model=GLNO2D_23_10_30,CNO2D_13_18_48,FNO2D_18_03_29,LNO2D_18_17_36,WNO2D_14_39_04 --model_list=GLNO,CNO,FNO,LNO,WNO

python visulize.py --task=beam --local_rank=2 --load_model=GLNO2D_17_52_44,CNO2D_12_41_37,FNO2D_17_33_06,LNO2D_17_57_56,WNO2D_14_03_03 --model_list=GLNO,CNO,FNO,LNO,WNO

python train.py --task=reacdiffusion --local_rank=1 --config=config_glno.yaml
python visulize.py --task=reacdiffusion --local_rank=2 --load_model=GLNO2D_17_35_50,FNO2D_18_23_20,LNO2D_17_07_13,WNO2D_14_32_38 --model_list=GLNO,FNO,LNO,WNO


python train.py --task=beam --local_rank=2 
python train.py --task=diffusion --local_rank=1 --config=config_glno.yaml

python train.py --task=shrec11_simplified --local_rank=1 --config=config_geofno.yaml

```

results are saved in logs/task/date/model_time/train.log

## visualization
private

haven't rewrite yet.


python visulize.py --task=turbulent --local_rank=2 --load_model=GLNO2D_14_11_58,FNO2D_15_43_36,LNO2D_15_49_12,WNO2D_21_50_04 --model_list=GLNO,FNO,LNO,WNO