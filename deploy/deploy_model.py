import os
import subprocess
import sys
from sagemaker.tensorflow import TensorFlowModel
import time
import sagemaker
from datetime import datetime
import pandas as pd


def upload_model_to_s3(local_model_path, s3_bucket_name, s3_model_path):
    if not os.path.exists(local_model_path):
        print(f"Erro: o arquivo local '{local_model_path}' não existe.")
        sys.exit(1)

    aws_cli_command = [
        'aws', 's3', 'cp', local_model_path,
        f's3://{s3_bucket_name}/{s3_model_path}'
    ]

    try:
        subprocess.run(aws_cli_command, check=True)
        print(f"Modelo enviado com sucesso para o S3: s3://{s3_bucket_name}/{s3_model_path}") # noqa
    except subprocess.CalledProcessError as e:
        print(f"Erro ao fazer upload do modelo: {e}")
        sys.exit(1)


def deploy_model_to_sagemaker(model_s3_path):
    role_arn = "arn:aws:iam::896684835246:role/vocareum"

    tensorflow_model = TensorFlowModel(
        model_data=model_s3_path,
        role=role_arn,
        framework_version='2.16.1'
    )

    try:
        print("Iniciando o deploy do modelo...")
        predictor = tensorflow_model.deploy(
            initial_instance_count=1,
            instance_type='ml.t2.medium'
        )
        print("Modelo implantado com sucesso.")

    except Exception as e:
        print(f"Erro durante o deploy: {e}")
        sys.exit(1)

    print("Model deployed successfully.")

    # Verificar o status do endpoint
    endpoint_name = predictor.endpoint_name
    print(f"Endpoint Name: {endpoint_name}")

    while True:
        try:
            response = sagemaker.Session().describe_endpoint(
                EndpointName=endpoint_name)
            status = response['EndpointStatus']
            print(f"Endpoint Status: {status}")
            if status == 'InService':
                print("Endpoint is in service.")
                break
            elif status == 'Failed':
                print("Endpoint creation failed.")
                break
        except Exception as e:
            print(f"Erro ao verificar status do endpoint: {e}")
            break
        time.sleep(30)

    test_data_df = pd.DataFrame({
        "predict_price": [10.3399181366, 10.3048620224, 10.3935070038,
                          10.3032579422, 10.0754566193]
    })
    response = predictor.predict(test_data_df)

    print("Prediction result:", response)

    predictor.delete_endpoint()


def deploy_model():
    today = datetime.today().strftime('%Y%m%d')

    # local_model_path = f'Models/{today}/ITSA4.keras'

    s3_bucket_name = 'tc4grupo46'
    s3_model_path = f'trainedModel/{today}/ITSA4.keras'

    # upload_model_to_s3(local_model_path, s3_bucket_name, s3_model_path)

    # Etapa 2: Realizar o deploy no SageMaker
    model_s3_path = f's3://{s3_bucket_name}/{s3_model_path}'
    deploy_model_to_sagemaker(model_s3_path)


if __name__ == "__main__":
    deploy_model()
