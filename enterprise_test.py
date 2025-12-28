from flask import Flask, request
import os
import pandas as pd

app = Flask(__name__)

# API ROUTE 1: Login
@app.route('/login', methods=['POST'])
def login_user():
    username = request.form['user']
    # SECURITY RISK: Hardcoded secret!
    password = "SuperSecretPassword123" 
    
    if check_db(username, password):
        return "Success"
    return "Fail"

# API ROUTE 2: Admin Panel
@app.route('/admin/execute')
def admin_exec():
    cmd = request.args.get('cmd')
    # SECURITY RISK: Dangerous Eval!
    eval(cmd) 
    return "Executed"

def check_db(u, p):
    # Uses pandas library
    df = pd.read_csv("users.csv")
    return True