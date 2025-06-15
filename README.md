(I) Datasets
The datasets used in our study can be accessed at:
https://doi.org/10.1038/s41592-022-01480-9

Please download and organize the datasets under a directory named ../DataUpload/, e.g., ../DataUpload/Dataset1/.

(II) Core Files
main.py
Main script for training and prediction. Accepts dataset path as a command-line argument.

MyModel.py
Contains the implementation of the GCNgene model architecture.

Calculate.py
Contains functions for computing evaluation metrics (e.g., RMSE, R², correlation).

(III)  How to Train and Predict
After preparing the dataset:

bash

python main.py --datapath="../DataUpload/Dataset1/"
This command will start training the model and perform prediction on the specified dataset.

(IV)  Contact
If you are interested in our work or have any suggestions and questions about our research work, please feel free to contact us. E-mail: yingzhang@njust.edu.cn.




