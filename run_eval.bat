SET PYTHONPATH=%cd%

SET MODEL_DIR=log/resface64/20201028-193142-multiscale
call :fun_eval_all



exit /b 0

:fun_eval_all
    echo %MODEL_DIR%

    SET TARGET=lfw,calfw,cplfw,cfp_ff,cfp_fp,agedb_30,vgg2_fp
    SET TARGET=lfw

    d:\Anaconda3\python.exe evaluation\verification.py ^
        --target %TARGET% ^
        --model_dir %MODEL_DIR%

    d:\Anaconda3\python.exe evaluation\verification_risk.py ^
        --target %TARGET% ^
        --model_dir %MODEL_DIR%

    d:\Anaconda3\python.exe evaluation\eval_lfw_openset.py ^
        --model_dir %MODEL_DIR% ^
        --dataset_path F:\data\face-recognition\lfw\lfw-112-mxnet

GOTO:EOF
