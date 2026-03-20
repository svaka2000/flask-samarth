from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource

from model.titanic import TitanicModel

titanic_api = Blueprint('titanic_api', __name__,
                        url_prefix='/api/titanic')

api = Api(titanic_api)

class TitanicAPI:
    class _Predict(Resource):

        def post(self):
            """ Semantics: In HTTP, POST requests are used to send data to the server for processing.
            Sending passenger data to the server to get a prediction fits the semantics of a POST request.

            POST requests send data in the body of the request...
            1. which can handle much larger amounts of data and data types, than URL parameters
            2. using an HTTPS request, the data is encrypted, making it more secure
            3. a JSON formated body is easy to read and write between JavaScript and Python, great for Postman testing
            """
            passenger = request.get_json()

            titanicModel = TitanicModel.get_instance()
            response = titanicModel.predict(passenger)

            return jsonify(response)

    api.add_resource(_Predict, '/predict')
