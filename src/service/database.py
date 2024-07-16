import os
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine

# 환경 변수 로드
load_dotenv()

# 환경 변수에서 데이터베이스 설정 가져오기
host = os.getenv('DB_HOST')
user = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD')
database = os.getenv('DB_NAME')

# SQLAlchemy 엔진 생성
engine = create_engine(f'mysql+mysqlconnector://{user}:{password}@{host}/{database}')


def developer_part_eq(part):
    query = "SELECT developer_id, short_intro FROM developer WHERE part1 = %s OR part2 = %s"
    df = pd.read_sql(query, engine, params=(part.value, part.value))
    return df
