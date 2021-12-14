from flask import Flask, request
import tensorflow_text as tf_text
import tensorflow as tf
import json
import requests

app = Flask(__name__)

lower_entities= ["hasstate", "ison", "isin", "devicename", "statevalue"]
upper_entities= ["hasState", "isOn", "isIn", "deviceName", "stateValue"]
translator = tf.saved_model.load('../saved_model/translator/')
url = "http://206.12.91.26:8070/CASM-2.0.1/api/query"
headers = {'Content-Type': 'text/plain'}

def get_result(query):
    response = requests.request("POST", url, headers=headers, data=query).json()["device"]
    result = response["results"]
    return result 

def decode_query(output):
    for i in range(len(lower_entities)):
        output = output.replace(lower_entities[i], upper_entities[i])
    return output

@app.route('/convert_text_to_cdql', methods=['POST'])
def convert_text_to_cdql():
    data = request.get_json()
    input_text = tf.constant([data["text"]])
    result = translator.tf_translate(input_text)['text'][0].numpy().decode()
    query = decode_query(result)
    return json.dumps({"cdql": query})


@app.route('/convert_cdql_to_results', methods=['POST'])
def convert_cdql_to_results():
    data = request.get_json()
    query = data["cdql"]
    result = get_result(query)
    return json.dumps({"results": str(result)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=8080)

