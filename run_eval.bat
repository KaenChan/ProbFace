SET PYTHONPATH=G:\chenkai\probface-github

SET MODEL_DIR=G:\chenkai\Probabilistic-Face-Embeddings-master\log\resface64_relu_msarcface_am_PFE\20191216-084820-tpr99.65
call :fun_eval_all



exit /b 0

:fun_eval_all
    echo %MODEL_DIR%

    SET TARGET=lfw,calfw,cplfw,cfp_ff,cfp_fp,agedb_30,vgg2_fp
    SET TARGET=lfw

    0d:\Anaconda3\python.exe evaluation\verification.py ^
        --target %TARGET% ^
        --model_dir %MODEL_DIR%

    d:\Anaconda3\python.exe evaluation\eval_lfw_openset.py ^
        --model_dir %MODEL_DIR% ^
        --dataset_path F:\data\face-recognition\lfw\lfw-112-mxnet

    move *.npy %MODEL_DIR%

GOTO:EOF
