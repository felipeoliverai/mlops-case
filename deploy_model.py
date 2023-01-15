import sagemaker as sage
from sagemaker.predictor import csv_serializer



def main(): 

    image_tag = 'aws-model-titanic' # use the <image_name> defined earlier
    sess = sage.Session()
    #role = get_execution_role()
    role = 'arn:aws:iam::064559986273:role/sagemaker-model-titanic-role'
    account = sess.boto_session.client('sts').get_caller_identity()['Account']
    region = sess.boto_session.region_name
    image = f'{account}.dkr.ecr.{region}.amazonaws.com/{image_tag}:latest'

    # this is just a dummy location. The model is called with train data. We use the current notebook as dummy train data.
    uri = sess.upload_data('./deploy_model.py')

    artifacts = 's3://model-artifact-titanic-1/artifact/'
    sm_model = sage.estimator.Estimator(image,
                                    role,
                                    1,
                                    'ml.c4.xlarge', output_path=artifacts, sagemaker_session=sess)

    # Run the train program because it is expected
    sm_model.fit(uri)

    # Deploy the model.
    predictor = sm_model.deploy(1, 'ml.m4.xlarge', serializer=csv_serializer)
    print("Deployed!!!!")
    print(predictor)


if __name__ == "__main__":
    main()