from sagemaker.sklearn import SKLearn
from sagemaker.sklearn.model import SKLearnModel
import sagemaker 
import boto3
import logging
import sagemaker as sage
from sagemaker import get_execution_role
from sagemaker.predictor import csv_serializer
import json



def main(): 

    # Creating a SageMaker runtime client
    runtime = boto3.client('runtime.sagemaker')

    # CSV input data
    payload_1 = '26,6.002,False,0,1,0,0,0'
    payload_2 = '26,6.002,False,0,1,0,1,1'

    # Endpoint name
    endpoint_name = 'aws-model-titanic-2023-01-15-17-25-58-708'

    # Perform inference
    response = runtime.invoke_endpoint(EndpointName=endpoint_name,
                                    ContentType='text/csv',
                                    Body=payload_1)

    # Get the result
    result = response['Body'].read()

    # Decode the result from binary to string
    result = result.decode("utf-8")

    # remove leading/trailing whitespaces
    result = result.strip()

    if result == '1': 
        print("Survived")
    else: 
        print("Not Survived")
        
    # Print the result
    print(result)


if __name__ == "__main__":
    main()