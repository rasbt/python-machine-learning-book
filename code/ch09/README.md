Sebastian Raschka, 2015

# Python Machine Learning
# Chapter 9 Code Examples

## Embedding a Machine Learning Model into a Web Application


The code for the Flask web applications can be found in the following directories:
    
- `1st_flask_app_1/`: A simple Flask web app
- `1st_flask_app_2/`: `1st_flask_app_1` extended with flexible form validation and rendering
- `movieclassifier/`: The movie classifier embedded in a web application
- `movieclassifier_with_update/`: same as `movieclassifier` but with update from sqlite database upon start


To run the web applications locally, `cd` into the respective directory (as listed above) and execute the main-application script, for example,

    cd ./1st_flask_app_1
    python3 app.py
    
Now, you should see something like
    
     * Running on http://127.0.0.1:5000/
     * Restarting with reloader
     
in your terminal.
Next, open a web browsert and enter the address displayed in your terminal (typically http://127.0.0.1:5000/) to view the web application.