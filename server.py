
import json
import os
import requests
import tensorflow as tf

from argparse import ArgumentParser
from flask import Flask, request, jsonify, render_template

from bert_classify import BERTClassifier

app = Flask(__name__)


def load_model(project):
	global bc
	global graph
	global session
	graph = tf.get_default_graph()
	session = tf.Session()
	with session.as_default():
		bc = BERTClassifier(project=project, mode='predict')


@app.route('/getintent', methods=['POST'])
def predict():
	data = {'success': False}
	if request.method == 'POST':
		print(request)
		response = request.data.decode('utf8')
		content = json.loads(response)
		text = content['text']
		with graph.as_default():
			with session.as_default():
				print(text, "=====")
				data['predictions'] = bc.predict(text=text)
				data['success'] = True
	return jsonify(data)


if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("-p", "--project", required=True,
						help="Enter the project name. Eg: `bank` or `imdb`")
	args = parser.parse_args()
	print(("* Loading BERT model and starting Flask Server..."
           "please wait until server has fully started"))
	load_model(project=args.project)
	app.run(threaded=False)

'''
curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"text": "Get me account summary"}' \
  http://localhost:5000/getintent
'''