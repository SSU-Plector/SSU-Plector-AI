from flask import Flask
from flask_restx import Api, Resource, fields

from src.enum.part import Part
from src.health_check import health_check_module
from src.service.developer_matching.matching import developer_matching

application = Flask(__name__)

api = Api(application, version='1.0.0', title='SSU-Plector AI', description='SSU-Plector AI API 문서', doc='/swagger-ui')

# api url
health_api = api.namespace('test', description='health check API')
ai_match_api = api.namespace('ai', description='AI Match API')

# dto
developer_matching_dto = api.model('DeveloperMatchingDTO', {
    'part': fields.String(required=True, description='part', enum =[e.value for e in Part]),
    'request': fields.String(required=False, description='request')
})


@health_api.route('/health')
class Test(Resource):
    def get(self):
        return health_check_module()


@ai_match_api.route('/developer_match')
class Match(Resource):
    @ai_match_api.expect(developer_matching_dto)
    def post(self):
        data = api.payload
        matched_data = developer_matching(data)

        return matched_data


if __name__ == '__main__':
    application.run()
