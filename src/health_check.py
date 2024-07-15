from flask import jsonify

def health_check_module():
    return jsonify(status="UP"), 200
