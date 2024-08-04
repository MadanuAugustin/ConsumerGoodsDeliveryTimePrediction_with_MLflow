## Predictive Modelling for Consumer Goods Delivery Time Prediction

This project focuses on using deep learning techniques to predict the delivery time of consumer goods. Accurate prediction of delivery times is crucial for optimizing logistics, enhancing customer satisfaction, and reducing operational costs. By employing machine learning algorithms, this project aims to predict delivery times accurately based on relevant features such as order details, shipping methods, and delivery locations.

### Objective:
The primary objective of this project is to develop reliable prediction models that accurately forecast the delivery time of consumer goods. These models will assist logistics companies in optimizing their delivery processes, ultimately leading to improved customer satisfaction and reduced operational costs.

### Dataset:
The project utilizes a dataset containing relevant attributes such as order details, shipping methods, and delivery locations. This dataset serves as the foundation for training and evaluating the deep learning models.

### Techniques:
Deep learning algorithms and techniques explored in this project include:
- Long Short Term Memory

### Evaluation:
The performance of the models is evaluated using metrics such as r2score, mean squared error, and mean absolute error. Techniques such as cross-validation and hyperparameter tuning are employed to enhance the models' reliability and applicability across different delivery conditions.

![image](https://github.com/user-attachments/assets/aa40381d-3f67-4d6e-8d34-36636c797536)

---

### PROJECT STRUCTURE:

![image](https://github.com/user-attachments/assets/4618f440-5a52-4e7d-874f-48602eddaf1a)

---

### FINAL OUTPUT:

https://github.com/user-attachments/assets/62c70b4d-836d-4068-a972-119c3ce711e7

---

### MLflow Experiments:
https://dagshub.com/augustin7766/ConsumerGoodsDeliveryTimePrediction_with_MLflow

## AWS-CICD-Deployment-with-Github-Actions:
### 1. Login to AWS console.
### 2. Create IAM (Identity Access Manager) user for deployment.

    1. EC2 access : It is virtual machine

    2. ECR: Elastic Container registry to save your docker image in aws

    #Description: About the deployment

    1. Build docker image of the source code

    2. Push your docker image to ECR

    3. Launch Your EC2 

    4. Pull Your image from ECR in EC2

    5. Lauch your docker image in EC2


    #Policy:

    1. AmazonEC2ContainerRegistryFullAccess

    2. AmazonEC2FullAccess

### 3. Create ECR repo to store/save docker image
    - Save the URI: 566373416292.dkr.ecr.us-east-1.amazonaws.com/semiconductor

### 4. Create EC2 machine (Ubuntu)

### 5. Open EC2 and Install docker in EC2 Machine:

    sudo apt-get update -y

    sudo apt-get upgrade

    #required

    curl -fsSL https://get.docker.com -o get-docker.sh

    sudo sh get-docker.sh

    sudo usermod -aG docker ubuntu

    newgrp docker

### 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one

### 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = us-east-1

    AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

    ECR_REPOSITORY_NAME = simple-app




