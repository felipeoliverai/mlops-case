from sagemaker.sklearn import SKLearn
from sagemaker.sklearn.model import SKLearnModel
import sagemaker 
import boto3
import logging



def main():
    

    role = 'sagemaker-model-titanic-role'

    # open SageMaker session 
    sagemaker_session = sagemaker.Session()
    inputs = sagemaker_session.upload_data(path="model/model.tar.gz", key_prefix="model")

    # bring own model 
    sagemaker_model = SKLearnModel(
        model_data="s3://model-artifact-titanic-1/artifact/model.tar.gz",
        role=role,
        framework_version='0.23-1',
        py_version='py3',
        entry_point="train.py",
    )

    print("Training Model Packaged!")

    # ml.m4.xlarge

    logging.getLogger().setLevel(logging.WARNING)
    predictor = sagemaker_model.deploy(initial_instance_count=1, instance_type="ml.m4.xlarge")
    print(predictor)
    print("\n Deployed!!!")

if __name__ == "__main__":
    main()