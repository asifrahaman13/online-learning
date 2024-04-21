# About the application.

First clone repository.

```bash
git clone http://github.com/asifrhaman13/ml_project
```

Create virtual environment.

```bash 
virtualenv .venv
```

Next activate the virtual environment.

```bash
source .venv/bin/activate
```

Install the dependencies.

```bash
pip instal -r requirements.txt
```

Run the script. 

```bash
uvicorn main:app --reload 
```

```bash 
chmod +x start.sh
bash start.sh
```

