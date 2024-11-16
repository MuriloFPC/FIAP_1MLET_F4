import os
from dotenv import load_dotenv

def GetEnvStocks():
    load_dotenv(dotenv_path='../.env')

    return os.getenv("STOCKS", "ITUB4").split(",")