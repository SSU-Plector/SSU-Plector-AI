from src.health_check import health_check_module  # 모듈의 상대 경로 임포트
from flask import Flask

application = Flask(__name__)

# Health check route
application.add_url_rule('/health', 'health_check', health_check_module)

if __name__ == '__main__':
    application.run()
