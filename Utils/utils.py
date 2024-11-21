import os
from dotenv import load_dotenv, set_key

def GetEnvStocks():
    load_dotenv(dotenv_path='../.env')

    return os.getenv("STOCKS", "ITUB4").split(",")

def GetEnvVariable(env):
    load_dotenv(dotenv_path='../.env')
    env_variable = os.getenv(env, None)
    if env_variable is None:
        raise KeyError(f'Environment variable {env} not found')
    return env_variable

def SaveEnvVariable(env, value):
    set_key('../.env', env, value)