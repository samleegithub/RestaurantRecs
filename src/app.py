from flask import Flask, request, render_template

@app.route('/recommend', methods=['POST'])
def recommend():
	json_doc = request.json

	return 'Hello world! Hoping to recommend stuff here'


@app.route('/')
def index():
	return render_template('index.html')


def main():
	app = Flask(__name__)

    # Start Flask app
    app.run(host='0.0.0.0', port=PORT, debug=True)


if __name__ == '__main__':
	main()