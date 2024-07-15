from flask import Flask
from flask_restx import Api, Resource

from src.health_check import health_check_module

application = Flask(__name__)

api = Api(application, version='1.0.0', title='SSU-Plector AI', description='SSU-Plector AI API 문서', doc='/swagger-ui')

health_api = api.namespace('test', description='health check API')
@health_api.route('/health')
class Test(Resource):
    def get(self):
        return health_check_module()


if __name__ == '__main__':
    application.run()
