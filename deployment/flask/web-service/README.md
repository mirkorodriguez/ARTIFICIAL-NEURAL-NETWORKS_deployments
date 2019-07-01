## Deploying on IBM Cloud
We should use the shell script deploy-on-ibmcloud.sh to deploy on IBM-Cloud.
Your IBM Account credentials will be needed.

```bash
 chmod +x deploy-on-ibmcloud.sh
 ./deploy-on-ibmcloud.sh
```

## Deploying locally
*Activate your environment. In my case it's called 'tensorflow'

```bash
> source activate tensorflow
> python service.py
  or  
> python3 service.py
```
*The port by default is 5000, so you should lauch the app on http://localhost:5000/


#Python runtime in IBM-Cloud:
https://devcenter.heroku.com/articles/python-runtimes
