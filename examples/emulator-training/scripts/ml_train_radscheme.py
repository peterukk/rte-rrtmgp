#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python framework for developing neural networks to replace radiative
transfer computations, either fully or just one component

This code is for emulating RTE+RRTMGP (entire radiation scheme)

This program takes existing input-output data generated with RTE+RRTMGP and
user-specified hyperparameters such as the number of neurons, optionally
scales the data, and trains a neural network. 

Temporary code

Contributions welcome!

@author: Peter Ukkonen
"""
import os
import gc
import numpy as np

from ml_loaddata import load_inp_outp_radscheme, preproc_minmax_inputs, \
    preproc_pow_gptnorm, preproc_pow_gptnorm_reverse
from ml_eval_funcs import plot_hist2d
import matplotlib.pyplot as plt

def calc_heatingrates(y, p):
    fluxup = y[:,0:61]
    fluxdn = y[:,61:]
    F = fluxdn - fluxup
    dF = np.gradient(F,axis=1)
    dp = np.gradient(p,axis=1)
    dFdp = dF/dp
    g = 9.81 # m s-2
    cp = 1004 # J K-1  kg-1
    dTdt = -(g/cp)*(dFdp) # K / s
    dTdt_day = (24*3600)*dTdt
    return dTdt_day

def rmse(predictions, targets,ax=0):
    return np.sqrt(((predictions - targets) ** 2).mean(axis=ax))

def mae(predictions,targets,ax=0):
    diff = predictions - targets
    return np.mean(np.abs(diff),axis=ax)

def plot_heatingrate_error(hr_true, hr_pred, pres):
    # errfunc = mae
    errfunc = rmse
    ind_p = 5
    hre_radscheme       = errfunc(hr_true[:,ind_p:], hr_pred[:,ind_p:])
    yy = 0.01*pres[:,:].mean(axis=0)
    figtitle = 'ERrror in shortwave heating rate'
    fig, ax = plt.subplots(1)
    ax.plot(hre_radscheme,  yy[ind_p:])
    ax.invert_yaxis(); ax.grid()
    ax.set_ylabel('Pressure (hPa)',fontsize=15)
    ax.set_xlabel('Heating rate (W m$^{-2}$)',fontsize=15); 
    fig.suptitle(figtitle, fontsize=16)
    
def plot_flux_and_hr_error(y_true, y_pred, pres):
    fluxup_true = y_true[:,0:61]; fluxdn_true = y_true[:,61:]
    fluxup_pred = y_pred[:,0:61]; fluxdn_pred = y_pred[:,61:]
    dF_true = np.gradient((fluxdn_true - fluxup_true),axis=1)
    dF_pred = np.gradient((fluxdn_pred - fluxup_pred),axis=1)
    dp = np.gradient(pres,axis=1)
    dFdp_true = dF_true/dp; dFdp_pred = dF_pred/dp
    g = 9.81; cp = 1004 
    dTdt_true = -(24*3600)*(g/cp)*(dFdp_true) # K / h
    dTdt_pred = -(24*3600)*(g/cp)*(dFdp_pred) # K / h
    errfunc = mae
    #errfunc = rmse
    ind_p = 5
    hr_err      = errfunc(dTdt_true[:,ind_p:], dTdt_pred[:,ind_p:])
    fluxup_err  = errfunc(fluxup_true[:,ind_p:], fluxup_pred[:,ind_p:])
    fluxdn_err  = errfunc(fluxdn_true[:,ind_p:], fluxdn_pred[:,ind_p:])
    fluxnet_err  = errfunc((fluxdn_true[:,ind_p:] - fluxup_true[:,ind_p:]), \
                           (fluxdn_pred[:,ind_p:] - fluxup_pred[:,ind_p:] ))
    yy = 0.01*pres[:,:].mean(axis=0)
    fig, (ax0,ax1) = plt.subplots(ncols=2, sharey=True)
    ax0.plot(hr_err,  yy[ind_p:], label='SW Heating rate error')
    ax0.invert_yaxis()
    ax0.set_ylabel('Pressure (hPa)',fontsize=15)
    ax0.set_xlabel('Heating rate (K h$^{-1}$)',fontsize=15); 
    ax1.set_xlabel('Flux (W m$^{-2}$)',fontsize=15); 
    ax1.plot(fluxup_err,  yy[ind_p:], label='SW upward flux error')
    ax1.plot(fluxdn_err,  yy[ind_p:], label='SW downward flux error')
    ax1.plot(fluxnet_err,  yy[ind_p:], label='SW net flux error')
    ax1.legend()

# ----------------------------------------------------------------------------
# ----------------- RTE+RRTMGP EMULATION  ------------------------
# ----------------------------------------------------------------------------

#  ----------------- File paths -----------------
datadir     = "/media/peter/samsung/data/CAMS/ml_training/"
datadir     = "/home/puk/data/"
fpath       = datadir + "/RADSCHEME_data_g224_CAMS_2009-2018_sans_2014-2015.nc"
fpath_val   = datadir + "/RADSCHEME_data_g224_CAMS_2014.nc"
fpath_test  = datadir +  "/RADSCHEME_data_g224_CAMS_2015.nc"

# fpath       = "/home/puk/soft/rte-rrtmgp-nn/examples/emulator-training/data_training/ml_data_g224_CAMS_2011-2013_clouds.nc"
# fpath_val   = "/home/puk/soft/rte-rrtmgp-nn/examples/emulator-training/data_training/ml_data_g224_CAMS_2018_clouds.nc"
# fpath_test   = "/home/puk/soft/rte-rrtmgp-nn/examples/emulator-training/data_training/ml_data_g224_CAMS_2018_clouds.nc"

# fpath_test = "/media/peter/samlinux/data/data_training/ml_data_g224_withclouds_CAMS_2011-2013_RFMIPstyle.nc"

# ----------- config ------------

scale_inputs    = True
scale_outputs   = True

# Which ML library to use: select either 'pytorch',
# or 'tf-keras' for Tensorflow with Keras frontend
# ml_library = 'pytorch'
ml_library = 'tf-keras'

# Model training: use GPU or CPU?
use_gpu = False

# Tune hyperparameters using KerasTuner?
tune_params = False

# ----------- config ------------

# Load data
x_tr_raw, y_tr_raw = load_inp_outp_radscheme(fpath, scale_p_h2o_o3 = scale_inputs)

if (fpath_val != None and fpath_test != None): # If val and test data exists
    x_val_raw, y_val_raw   = load_inp_outp_radscheme(fpath_val, scale_p_h2o_o3 = scale_inputs)
    x_test_raw,y_test_raw, pres_test  = load_inp_outp_radscheme(fpath_test, \
                            scale_p_h2o_o3 = scale_inputs, return_pressures=True)
else: # if we only have one dataset, split manually
    from sklearn.model_selection import train_test_split
    train_ratio = 0.70
    validation_ratio = 0.15
    test_ratio = 0.15
    testval_ratio = test_ratio/(test_ratio + validation_ratio)
    # first split into two, training and test+val
    x_tr_raw, x_test_raw, y_tr_raw, y_test_raw = \
        train_test_split(x_tr_raw, y_tr_raw, test_size=1-train_ratio)
    # then split the latter into to testing and val
    x_val_raw, x_test_raw, y_val_raw, y_test_raw = \
        train_test_split(x_test_raw, y_test_raw, test_size=testval_ratio) 

# Number of inputs and outputs    
nx = x_tr_raw.shape[1]
ny = y_tr_raw.shape[1]   


if scale_inputs:
    x_tr        = np.copy(x_tr_raw)
    x_val       = np.copy(x_val_raw)
    x_test      = np.copy(x_test_raw)
    
    fpath_xcoeffs = "../../../neural/data/nn_radscheme_xmin_xmax.txt"
    xcoeffs = np.loadtxt(fpath_xcoeffs, delimiter=',')
    xmax = xcoeffs[0:542]

    # xmax = xcoeffs[542:]
    # xmin = xcoeffs[0:542]
    xmin = np.repeat(0.0, 542)

    
    x_tr            = preproc_minmax_inputs(x_tr_raw, (xmin, xmax))
    # x_tr, xmin,xmax = preproc_minmax_inputs(x_tr_raw)
    x_val           = preproc_minmax_inputs(x_val_raw,  (xmin,xmax)) 
    x_test          = preproc_minmax_inputs(x_test_raw, (xmin,xmax)) 
else:
    x_tr    = x_tr_raw
    x_val   = x_val_raw
    x_test  = x_test_raw
    
    
if scale_outputs: 
    # y_mean = np.zeros(ny)
    # y_sigma = np.zeros(ny)
    # for i in range(ny):
    #     y_mean[i] = y_tr_raw[:,i].mean()
    #     # y_sigma[igpt] = y_raw[:,igpt].std()
    # # y_mean = np.repeat(y_raw.mean(),ny)
    # y_sigma = np.repeat(y_tr_raw.std(),ny)  # 467.72
    # y_mean = np.array([374.46596068, 374.45956871, 374.45462027, 374.4511992 ,
    #    374.45047469, 374.45502195, 374.4680579 , 374.49239519,
    #    374.53055628, 374.58516238, 374.65892284, 374.75524611,
    #    374.8792801 , 375.03542025, 375.22643646, 375.45378099,
    #    375.71533287, 376.00285752, 376.30151249, 376.58998858,
    #    376.84997017, 377.06088253, 377.1959924 , 377.2298344 ,
    #    377.14753112, 376.9435507 , 376.61342392, 376.0964268 ,
    #    375.33952337, 374.26344007, 372.82036231, 370.98542669,
    #    368.80749149, 366.3915064 , 363.87406595, 361.26581523,
    #    358.59414259, 355.66949237, 352.05744001, 347.27688142,
    #    341.25040552, 334.76377694, 328.82791973, 323.08134132,
    #    316.79175838, 309.09519245, 299.90917436, 288.99921467,
    #    274.59731862, 257.36481124, 240.59263275, 226.58893258,
    #    216.13279391, 208.64963063, 203.82791957, 200.9401672 ,
    #    199.41826418, 198.6591551 , 198.25545666, 197.94484266,
    #    197.83066985, 897.8497014 , 897.56832421, 897.18702832,
    #    896.65574775, 895.92692424, 895.01315111, 893.947432  ,
    #    892.81558392, 891.67935113, 890.56589866, 889.47075025,
    #    888.3641525 , 887.21249947, 885.9978321 , 884.72128377,
    #    883.3744413 , 881.95796854, 880.47806976, 878.94735622,
    #    877.38503736, 875.785125  , 874.16277448, 872.56365052,
    #    871.02421366, 869.54206684, 868.07279715, 866.51043363,
    #    864.7201463 , 862.55637703, 859.83278134, 856.37222482,
    #    852.03649377, 846.8212828 , 840.8255009 , 834.20929904,
    #    826.98212955, 819.20462721, 810.68386777, 800.99420578,
    #    789.71007101, 776.9060625 , 763.57679203, 750.85167967,
    #    738.33882549, 725.31670556, 710.885572  , 695.07722868,
    #    677.67695876, 656.78869705, 633.36680778, 611.07829043,
    #    592.37422528, 578.04482565, 567.34107156, 559.94750542,
    #    555.05698858, 552.03360581, 550.17168289, 548.97796299,
    #    548.11269961, 547.68542119], dtype=np.float32)
    # y_sigma = np.repeat(431.14175665, ny)

    # nfac = 1
    # y_tr    = preproc_pow_gptnorm(y_tr_raw, nfac, y_mean, y_sigma)
    # y_val   = preproc_pow_gptnorm(y_val_raw, nfac, y_mean, y_sigma)
    # y_test  = preproc_pow_gptnorm(y_test_raw, nfac, y_mean, y_sigma)
    
    
    # ymax = np.array([1106.67077637, 1106.67883301, 1106.69067383, 1106.71191406,
    #    1106.74243164, 1106.78503418, 1106.8527832 , 1106.95166016,
    #    1107.08068848, 1107.23999023, 1107.43652344, 1107.68518066,
    #    1107.99768066, 1108.40307617, 1108.90881348, 1109.50048828,
    #    1110.1505127 , 1110.86206055, 1111.68774414, 1112.68432617,
    #    1113.90551758, 1115.41320801, 1117.12915039, 1118.87609863,
    #    1120.45800781, 1121.84082031, 1123.11462402, 1142.57910156,
    #    1161.6862793 , 1171.68432617, 1167.14025879, 1180.18505859,
    #    1176.45690918, 1215.22619629, 1231.7310791 , 1186.79760742,
    #    1158.34008789, 1132.04345703, 1132.84082031, 1133.69396973,
    #    1134.60595703, 1135.56274414, 1136.55029297, 1137.56518555,
    #    1138.61450195, 1139.70458984, 1141.68896484, 1144.60693359,
    #    1146.64282227, 1150.00219727, 1154.28479004, 1159.4420166 ,
    #    1163.51220703, 1166.53942871, 1169.04846191, 1171.00195312,
    #    1172.51855469, 1173.93200684, 1175.37036133, 1176.5057373 ,
    #    1177.14978027, 1412.        , 1411.80639648, 1411.45690918,
    #    1410.92602539, 1410.22106934, 1409.41320801, 1408.20141602,
    #    1406.98095703, 1405.66809082, 1404.35900879, 1403.08483887,
    #    1401.98425293, 1400.96813965, 1399.96374512, 1398.90148926,
    #    1397.67907715, 1396.29101562, 1394.9140625 , 1393.57666016,
    #    1391.9921875 , 1390.19213867, 1388.72570801, 1388.23840332,
    #    1387.67346191, 1387.10205078, 1397.42102051, 1410.49414062,
    #    1428.5078125 , 1425.93115234, 1424.50305176, 1421.10839844,
    #    1405.4251709 , 1410.58605957, 1431.6817627 , 1401.53491211,
    #    1397.09655762, 1390.52099609, 1388.16149902, 1387.62597656,
    #    1388.03027344, 1387.17871094, 1384.89868164, 1383.89562988,
    #    1384.18164062, 1384.18481445, 1383.9173584 , 1383.35375977,
    #    1382.60229492, 1382.34399414, 1382.96887207, 1383.37194824,
    #    1383.5723877 , 1383.6027832 , 1383.47607422, 1383.45092773,
    #    1383.88903809, 1384.18310547, 1384.43457031, 1384.64086914,
    #    1384.79309082, 1384.87719727])
    # ymin = np.repeat(0.0, ny)


    # y_tr            = preproc_minmax_inputs(y_tr_raw, (ymin, ymax))
    # # y_tr, ymin,ymax = preproc_minmax_inputs(y_tr_raw)
    # y_val           = preproc_minmax_inputs(y_val_raw,  (ymin,ymax)) 
    # y_test          = preproc_minmax_inputs(y_test_raw, (ymin,ymax)) 
    
    
    y_tr    = y_tr_raw / np.repeat(y_tr_raw[:,61].reshape(-1,1),y_tr_raw.shape[1], axis=1)
    y_val   = y_val_raw / np.repeat(y_val_raw[:,61].reshape(-1,1),y_val_raw.shape[1], axis=1)
    y_test  = y_test_raw / np.repeat(y_test_raw[:,61].reshape(-1,1),y_test_raw.shape[1], axis=1)
    
    rldm = 1431.6817626953
    x_tr = np.hstack((x_tr, y_tr_raw[:,61].reshape(-1,1)/rldm))
    x_val = np.hstack((x_val,y_val_raw[:,61].reshape(-1,1)/rldm))
    x_test = np.hstack((x_test,y_test_raw[:,61].reshape(-1,1)/rldm))
    nx = nx + 1
    
else:
    y_tr    = y_tr_raw    
    y_val   = y_val_raw
    y_test  = y_test_raw
    
    
hre_loss = True
if hre_loss:
    hre_tr      = calc_heatingrates(y_tr_raw,pres_tr)
    hre_val     = calc_heatingrates(y_val_raw,pres_tr)
    hre_test    = calc_heatingrates(y_test_raw,pres_test)
    nlev = np.int(ny/2)
    y_sigma_hr = np.std(hre_tr.flatten())
    y_sigma_hr = np.repeat(6.29414, nlev)
    y_mean_hr = np.zeros(nlev)
    for i in range(nlev):
        y_mean_hr[i] = hre_tr[:,i].mean()
    
    hre_tr_sc = preproc_pow_gptnorm(hre_tr, 1, y_mean_hr, y_sigma_hr)
    

gc.collect()
# Ready for training

# PYTORCH TRAINING
if (ml_library=='pytorch'):
    from torch import nn
    import torch
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, TensorDataset
    from ml_trainfuncs_pytorch import MLP#, MLP_cpu
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    
    lr          = 0.001
    batch_size  = 512
    nneur       = 256
    mymodel = nn.Sequential(
          nn.Linear(nx, nneur),
          nn.Softsign(), # first hidden layer
          nn.Linear(nneur, nneur),
          nn.Softsign(), # second hidden layer
          nn.Linear(nneur, ny) # output layer
        )
    
    x_tr_torch = torch.from_numpy(x_tr); y_tr_torch = torch.from_numpy(y_tr)
    data_tr  =  TensorDataset(x_tr_torch,y_tr_torch)
    
    x_val_torch = torch.from_numpy(x_val); y_val_torch = torch.from_numpy(y_val)
    data_val    = TensorDataset(x_val_torch,y_val_torch)
    
    x_test_torch = torch.from_numpy(x_test); y_test_torch = torch.from_numpy(y_test)
    data_test    = TensorDataset(x_test_torch,y_test_torch)
    
    mlp = MLP(nx=nx,ny=ny,learning_rate=lr,SequentialModel=mymodel)


    mc = pl.callbacks.ModelCheckpoint(monitor='val_loss',every_n_epochs=2)
    
    if use_gpu:
        trainer = pl.Trainer(gpus=0, deterministic=True)
    else:
        num_cpu_threads = 8
        trainer = pl.Trainer(accelerator="ddp_cpu", callbacks=[mc], deterministic=True,
                num_processes=  num_cpu_threads) 
                #plugins=pl.plugins.DDPPlugin(find_unused_parameters=False))
    
    # START TRAINING
    trainer.fit(mlp, train_dataloader=DataLoader(data_tr,batch_size=batch_size), 
            val_dataloaders=DataLoader(data_val,batch_size=batch_size))

    # PREDICT OUTPUTS FOR TEST DATA
    def eval_valdata():
        y_pred = mlp(x_val_torch)
        y_pred = y_pred.detach().numpy()
        # np.corrcoef(y_test.flatten(),y_pred.flatten())
        y_pred   = preproc_pow_gptnorm_reverse(y_pred, nfac, y_mean,y_sigma)
        
        plot_hist2d(y_test_raw,y_pred,20,True)      #  
    
    eval_valdata()


# TENSORFLOW-KERAS TRAINING
elif (ml_library=='tf-keras'):
    if tune_params:
        import optuna
    import tensorflow as tf
    from tensorflow.keras import losses, optimizers, layers
    from tensorflow.keras.callbacks import EarlyStopping
    from ml_trainfuncs_keras import create_model_mlp, savemodel
    import tensorflow.keras.backend as K

    mymetrics   = ['mean_absolute_error']
    valfunc     = 'val_mean_absolute_error'
    
    # Model architecture
    # First hidden layer (input layer) activation
    activ0      = 'softsign'
    # activ0       = 'relu'
    # Activation in other hidden layers
    activ       =  activ0    
    # Activation for last layer
    activ_last   = 'linear'
    # activ_last   = 'relu'
    activ_last   = 'sigmoid'

    epochs      = 100000
    patience    = 25
    lossfunc    = losses.mean_squared_error
    ninputs     = x_tr.shape[1]
    # lr          = 0.001
    lr          = 0.0001 
    # lr          = 0.0002 
    batch_size  = 256
    batch_size  = 1024
    neurons     = [182, 182]
    # neurons     = [256,256] #0.994935
    neurons     = [128, 128]  #0.99804565
    neurons     = [64, 64]  # 0.9952047
    neurons     = [128]     # 0.9980413605
                    # 0.9996379952819034          
    if use_gpu:
        devstr = '/gpu:0'
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    else:
        num_cpu_threads = 4
        devstr = '/cpu:0'
        # Maximum number of threads to use for OpenMP parallel regions.
        os.environ["OMP_NUM_THREADS"] = str(num_cpu_threads)
        # Without setting below 2 environment variables, it didn't work for me. Thanks to @cjw85 
        os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_cpu_threads)
        os.environ["TF_NUM_INTEROP_THREADS"] = str(1)
        os.environ['KMP_BLOCKTIME'] = '1' 

        tf.config.threading.set_intra_op_parallelism_threads(
            num_cpu_threads
        )
        tf.config.threading.set_inter_op_parallelism_threads(
            1
        )
        tf.config.set_soft_device_placement(True)
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    batch_size  = 2048
    lr          = 0.001 
    
    optim = optimizers.Adam(learning_rate=lr)
    
    if hre_loss:
        
        def custom_loss_wrapper(input_tensor):

            def custom_loss(y_true, y_pred):
                err_flux = K.sqrt(K.mean(K.square(y_true - y_pred),axis=0))
                
                # HR_true = layers.Lambda(lambda x: x[:,542]) * y_true
                # HR_pred = layers.Lambda(lambda x: x[:,542]) * y_pred

                # HR_true = input_tensor[:-1,542] * y_true
                # HR_pred = input_tensor[:-1,542] * y_pred
                
                # HR_true = tf.Variable(input_tensor[:-1,542] )
                HR_true = K.mean(input_tensor)

                # err_hr = K.sqrt(K.mean(K.square(HR_true - HR_pred),axis=0))
                # alpha = 0.1
                # err = (1-alpha) * err_hr + (alpha)*err_flux
                return err_flux*HR_true
            return custom_loss
        
        input_tensor = tf.keras.Input(shape=(nx,))

        def create_model_func(nx,ny,neurons, activ0, activ, activ_last):
            dense = layers.Dense(neurons[0], activation=activ0)
            x = dense(input_tensor)
            # further hidden layers
            for i in range(1,np.size(neurons)):
                x = layers.Dense(neurons[i], activation=activ)(x)
            # output layer
            outputs = layers.Dense(ny, activation=activ_last)(x)
            model = tf.keras.Model(inputs=input_tensor, outputs=outputs, name="mnist_model")

            return model
        
        model = create_model_func(nx=nx,ny=ny,neurons=neurons,activ0=activ0,activ=activ,
                         activ_last = activ_last)
    
        model.compile(loss=custom_loss_wrapper(input_tensor), optimizer='adam')
        
        history = model.fit(x_tr, y_tr, epochs= epochs, batch_size=batch_size, shuffle=True)
        
        
        inp = tf.keras.Input(shape=(nx,))
        def CustomLoss(y_true, y_pred, input_tensor):
            err_flux = K.sqrt(K.mean(K.square(y_true - y_pred)))
            
            HR_true = input_tensor[:-1,542] * y_true
            HR_pred = input_tensor[:-1,542] * y_pred
            err_hr = K.sqrt(K.mean(K.square(HR_true - HR_pred)))
            
            alpha = 0.1
            err = (1-alpha) * err_hr + (alpha)*err_flux
                
            return err_flux
        
        dense = layers.Dense(neurons[0], activation=activ0)
        x = dense(inp)
        out = layers.Dense(ny, activation=activ_last)(x)
        target = tf.keras.Input((ny,))
        model = tf.keras.Model(inputs=[inp,target], outputs=out)
        model.add_loss(CustomLoss(target,out,inp))
        model.compile(loss=None, optimizer='adam')
        model.fit(x=[x_tr,y_tr],y=None)

    # Create and compile model
    if tune_params:
        # 3. Create a study object and optimize the objective function.
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)
    else:
        # model = create_model_mlp(nx=nx,ny=ny,neurons=neurons,activ0=activ0,activ=activ,
        #                          activ_last = activ_last, kernel_init='he_uniform')
        model = create_model_mlp(nx=nx,ny=ny,neurons=neurons,activ0=activ0,activ=activ,
                                 activ_last = activ_last, kernel_init='lecun_uniform')
        
        model.compile(loss=lossfunc, optimizer=optim, metrics=mymetrics)
    model.summary()
    
    # Create earlystopper and possibly other callbacks
    earlystopper = EarlyStopping(monitor=valfunc,  patience=patience, verbose=1, mode='min',restore_best_weights=True)
    callbacks = [earlystopper]
    
    
    # START TRAINING
    with tf.device(devstr):
        history = model.fit(x_tr, y_tr, epochs= epochs, batch_size=batch_size, shuffle=True,  verbose=1, 
                            validation_data=(x_val,y_val), callbacks=callbacks)    
        

    # TEST
    y_pred      = model.predict(x_test);  
    # y_pred      = preproc_pow_gptnorm_reverse(y_pred, nfac, y_mean,y_sigma)
    # y_pred = preproc_minmax_reverse(y_pred, (ymin,ymax))
    y_pred = y_pred * np.repeat(y_test_raw[:,61].reshape(-1,1),y_test_raw.shape[1], axis=1)

    
    cc = np.corrcoef(y_test_raw.flatten(), y_pred.flatten())
    diff = np.abs(y_test_raw-y_pred)
    rmse_err = np.sqrt(((y_pred - y_test_raw) ** 2).mean())
    print("r {} max diff {} RMSE {}".format(cc[0,1],np.max(diff), rmse_err))
    
    plot_hist2d(y_test_raw,y_pred,20,True)      # 
    
    plot_flux_and_hr_error(y_test_raw, y_pred, pres_test)

    
    # SAVE MODEL
    kerasfile = "../../../neural/data/radscheme-128-128_incfluxnorm.h5"

    savemodel(kerasfile, model)
    
    from tensorflow.keras.models import load_model
    kerasfile = "s/media/peter/samlinux/gdrive/phd/soft/rte-rrtmgp-nn/neural/data/radscheme-128.h5"
    model = tf.lite.TFLiteConverter.from_keras_model(kerasfile)
    model = load_model(kerasfile,compile=False)
    
