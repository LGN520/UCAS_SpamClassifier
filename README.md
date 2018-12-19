# SpamClassifier
Spam classifier implemented by Spark MLLib

- Prerequisite:
	+ Python 3 (mine is 3.5.2)
	+ Pyspark 2.4.0 (supported in the virtual env 'spamenv')

- Get started:
    + Use ```git clone``` to get the project on your local machine
        * 'master' branch is the stable version and 'dev' branch is the developing version, please dont push them at any time
        * After clone the project, use ```git checkout -b dev-username``` to create your own branch just like existing branch 'dev-ssy'
        * You should do everything on your own branch, never merge your branch into 'master' or 'dev' directly
        * Remember to pull 'dev' before coding in your branch each time, if there is any updation just merge it into you own branch
    + You can use your own python runtime environment which must satisfy the above prerequisite, or you can use the virtual environment we support
        * Make sure you've installed pip or just use ```apt-get install python-pip```
        * Method 1:
            - Open spamenv/bin/activate, replace the VIRTUAL_ENV variable with your own path of the 'spamenv' directory
            - Use ```source spamenv/bin/activate``` to activate the virtual env
            - Use ```deactivate``` to exit the virtual env
            - Use ```python manage.py runserver``` to create Django server, then you can do some prediction via your browser through ```127.0.0.1:8000/polls/```
        * Method 2:
            - Juse use ```source launch``` to activate virtual env and create Django server
