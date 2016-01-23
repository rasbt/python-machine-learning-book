Sebastian Raschka, 2015

Python Machine Learning - Code Examples

## Chapter 9 - Embedding a Machine Learning Model into a Web Application

- Serializing fitted scikit-learn estimators
- Setting up a SQLite database for data storage
- Developing a web application with Flask
- Our first Flask web application
  - Form validation and rendering
  - Turning the movie classifier into a web application
- Deploying the web application to a public server
  - Updating the movie review classifier
- Summary

---

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
Next, open a web browser and enter the address displayed in your terminal (typically http://127.0.0.1:5000/) to view the web application.


**Link to a live example application built with this tutorial: http://raschkas.pythonanywhere.com/**.
