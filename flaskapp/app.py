from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def sample():
    if request.method == "GET":
        return render_template("index.html")
    
    img = request.files["image"]
    
    #TODO : implement predictions here
    prediction = "no disease"
    response = {"prediction": prediction}
    
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)