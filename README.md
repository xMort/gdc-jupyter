# GoodData Data Science Jupyter Notebooks

A collection of Jupyter notebooks that serve as a template for data science
related tasks.
                 
## Set up Jupyter server

Checkout the repository source code.

```  
git clone git@github.com:xMort/gdc-jupyter.git
cd gdc-jupyter
```

Create `.env` file (`touch .env`) and use it to set up environment with your Tiger API token in the following way: 

```
TOKEN=<your Tiger API token>
```
                                                                                                                  
Run the following commands to build the environment:
                                                    
```
make clean
make dev
source .venv/bin/activate
pip install ipykernel
python -m ipykernel install --name="gooddata" --user                                                     
```
   
Install Jupyter if you don't have it yet. Setup [brew](https://brew.sh/) first if you don't have it yet.

```                                          
brew install jupyter
```
    
Setup Jupyter server config by creating a new file `jupyter_server_config.py` in `~/.jupyter/`.
You will find the content of the file in the appendix of this readme.

Now you can finally start the Notebook server.

```
jupyter notebook
```

You should be able to access the notebooks at http://127.0.0.1:8686/jupyter/tree?token=60c1661cc408f978c309d04157af55c9588ff9557c9380e4fb50785750703da6. 

The notebooks should be available from dashboards with enabled notebook support as well now.
               
## Appendix


*jupyter_server_config.py*
```
"""Configuration for the Jupyter development server."""

import os

#################
# Logging
#################

c.ServerApp.log_level = 'INFO'

#################
# Network
#################

c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8686
c.ServerApp.port_retries = 0

#################
# Browser
#################

c.ServerApp.open_browser = False

#################
# Terminal
#################

c.ServerApp.terminals_enabled = True

#################
# Authentication
#################

c.IdentityProvider.token = '60c1661cc408f978c309d04157af55c9588ff9557c9380e4fb50785750703da6'

#################
# Security
#################

c.ServerApp.disable_check_xsrf = True
# ORIGIN = 'http://localhost:3208'
ORIGIN = '*'
# c.ServerApp.allow_origin = ORIGIN
c.ServerApp.allow_origin_pat = '.*'
c.ServerApp.allow_credentials = True
c.ServerApp.tornado_settings = {
  'headers': {
    'Access-Control-Allow-Origin': ORIGIN,
    'Access-Control-Allow-Methods': '*',
    'Access-Control-Allow-Headers': 'Accept, Accept-Encoding, Accept-Language, Authorization, Cache-Control, Connection, Content-Type, Host, Origin, Pragma, Referer, sec-ch-ua, sec-ch-ua-mobile, sec-ch-ua-platform, Sec-Fetch-Dest, Sec-Fetch-Mode, Sec-Fetch-Site, Upgrade, User-Agent, X-XSRFToken, X-Datalayer, Expires',
    'Access-Control-Allow-Credentials': 'true',
    'Content-Security-Policy': f"frame-ancestors 'self' {ORIGIN} ",
  },
  'cookie_options': {
    'SameSite': 'None',
    'Secure': True
  }
}
c.IdentityProvider.cookie_options = {
  "SameSite": "None",
  "Secure": True,
}

#################
# Server Extensions
#################

c.ServerApp.jpserver_extensions = {
    'jupyterlab': False,
}

#################
# Content
#################

c.ServerApp.root_dir = os.getcwd() + '/templates'

#################
# URLs
#################

c.ServerApp.base_url = '/jupyter'
c.ServerApp.default_url = '/jupyter/lab'

#################
# Kernel
#################

# See
# https://github.com/jupyterlab/jupyterlab/pull/11841
# https://github.com/jupyter-server/jupyter_server/pull/657
c.ZMQChannelsWebsocketConnection.kernel_ws_protocol = None # None or ''

#################
# JupyterLab
#################

c.LabApp.collaborative = True
```
