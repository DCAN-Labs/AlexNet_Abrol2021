Motion QC Score
===================

Model 3
-------

![Actual QC motion score vs. frequency of prediction](./img/qc_motion_score_prediction_model_03.png)

* Standardized RMSE (on validation set): **0.3486471956656407**
* Mean squared error (on validation set): 0.1235
* Model architecture: AlexNet3D_Dropout_Regression
* Optimizer: Adam
* Epochs: 1000
* Location: */home/feczk001/shared/data/AlexNet/MotionQC/model03.pt*

Model 8
-------

![Actual QC motion score vs. frequency of prediction](./img/qc_motion_score_prediction_model_08.png)

* Mean squared error (on eLabe): **0.2150**
* Model architecture: AdamNet
* Optimizer: Adam
* Epochs: 128
* Location: */home/feczk001/shared/data/AlexNet/MotionQC/model08.pt*
* Training data: BCP
* Test data: eLabe

Model 2
-------

* Mean squared error (on validation set): **0.4121**
* Model architecture: AlexNet3D_Dropout_Regression
* Optimizer: Adam
* Epochs: 100

Model 1
-------

* Mean squared error (on validation set): **0.4855**
* Model architecture: AlexNet3D_Dropout_Regression
* Optimizer: Adam
* Epochs: 55

Model 7
-------

![Actual QC motion score vs. frequency of prediction](./img/qc_motion_score_prediction_model_07.png)

* Standardized RMSE (on validation set): **1.08185627460534**
* Mean squared error (on validation set): 0.4542
* Mean squared error (on training set):   0.4139
* Model architecture: AlexNet3D_Dropout_Regression
* Optimizer: Adam
* Epochs: 128
* Dataset(s): BCP and eLabe
* Location: */home/feczk001/shared/data/AlexNet/MotionQC/model07.pt*

Model 6
-------

![Actual QC motion score vs. frequency of prediction](./img/qc_motion_score_prediction_model_06.png)

* Standardized RMSE (on validation set): **1.3027799144029593**
* Mean squared error (on validation set): 0.4758
* Model architecture: AlexNet3D_Dropout_Regression
* Optimizer: Adam
* Epochs: 64
* Dataset(s): BCP and eLabe
* Location: */home/feczk001/shared/data/AlexNet/MotionQC/model06.pt*

Model 5
-------

* Standardized RMSE (on validation set): **[undefined---sigma was zero**
* Mean squared error (on validation set): 0.3301
* Model architecture: AlexNet3D_Dropout_Regression
* Optimizer: SGD
* Epochs: 1000
* Location: */home/feczk001/shared/data/AlexNet/model05.pt*
