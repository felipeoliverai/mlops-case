import boto3 
from sagemaker import image_uris
import datetime
from time import gmtime, strftime
import datetime
import sagemaker
from sagemaker.sklearn.model import SKLearnModel




def main():

    # Role to give SageMaker permission to access AWS services.
    role= "arn:aws:iam::064559986273:role/sagemaker-model-titanic-role"

    # Create a SageMaker session
    sagemaker_session = sagemaker.Session()

    # Define model artifact location in S3
    model_artifact = 's3://model-artifact-titanic-1/artifact/model.tar.gz'

    # Create a SKLearnModel
    model = SKLearnModel(model_data=model_artifact, 
                     role=role,
                     image_uri='064559986273.dkr.ecr.us-east-1.amazonaws.com/aws-model:latest',
                     entry_point='entry_point.py')


    # Deploy the model to an endpoint
    predictor = model.deploy(initial_instance_count=5, instance_type='ml.m5.xlarge', content_type='text/csv', endpoint_name='model-titanic-custom-case-v1')
    print(predictor)



if __name__ == "__main__":
    main() 
