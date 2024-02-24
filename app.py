from flask import Flask,jsonify,render_template,url_for,request,json
from utils import get_predicted_strength
import numpy as np
from config import PORT_NUMBER 



app = Flask(__name__)


@app.route("/home",methods = ["GET","POST"])
def home():

    return render_template("index.html")

@app.route("/predicted_stength",methods = ["GET","POST"])
def concrete_strength():
    result = None
    if request.method == "POST":
      data = request.form
      Cement = eval(data["Cement"])
      Blast_Furnace_Slag = eval(data["Blast_Furnace_Slag"])
      Fly_Ash = eval(data["Fly_Ash"])
      Water = eval(data["Water"])
      Superplasticizer = eval(data["Superplasticizer"])
      Coarse_Aggregate = eval(data["Coarse_Aggregate"])
      Fine_Aggregate = eval(data["Fine_Aggregate"])
      Age = eval(data["Age"])


      result = get_predicted_strength(Cement, Blast_Furnace_Slag, Fly_Ash, Water, Superplasticizer, Coarse_Aggregate, Fine_Aggregate, Age)
      print("predicted concrete strength = ",result)

    return render_template("index.html",prediction = result)



if __name__ =="__main__":
    app.run(host="0.0.0.0",port =PORT_NUMBER,debug = False )

