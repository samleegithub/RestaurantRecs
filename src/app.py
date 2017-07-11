from flask import Flask, request, render_template

app = Flask(__name__)
PORT = 5353

@app.route('/recommend', methods=['POST'])
def recommend():
    json_doc = request.json

    return 'Hello world! Hoping to recommend stuff here'


@app.route('/')
def index():
    return render_template('index.html')


def main():
    # Start Flask app
    app.run(host='0.0.0.0', port=PORT, debug=True)


if __name__ == '__main__':
    main()
