# SRL Models

## Available models
- autoencoder: an autoencoder from the raw pixels
- ground_truth: the agent's position
- srl_priors: SRL priors model
- supervised: a supervised model from the raw pixels to the agent's position
- pca: pca applied to the raw pixels
- vae: a variational autoencoder from the raw pixels
- joints: the arm's joints angles (kuka environments only)
- joints_position: the arm's x,y,z position and joints angles (kuka environments only)

## Add your own
If your SRL model is a carateristic of the environment (position, angles, ...):  
1. Add the name of the model to the ```registered_srl``` dictionary in ```state_representation/registry.py```, 
using this format ```NAME: (SRLType.ENVIRONMENT, [LIMITED_TO_ENV])```, where:
    * ```NAME```: is your model's name.
    * ```[LIMITED_TO_ENV]```: is the list of environment where this model works (will check for subclass), 
    set to ```None``` if this model applies to every environment.
2. Modifiy the ```def getSRLState(self, observation)``` in the environments to return the data you want for this model.  

Otherwise, for the SRL model that are external to the environment (Supervised, autoencoder, ...): 
1. Add your SRL model that inherits ```SRLBaseClass```, to the function ```state_representation.models.loadSRLModel```.
2. Add the name of the model to the ```registered_srl``` dictionary in ```state_representation/registry.py```, 
using this format ```NAME: (SRLType.SRL, [LIMITED_TO_ENV])```, where:
    * ```NAME```: is your model's name.
    * ```[LIMITED_TO_ENV]```: is the list of environment where this model works (will check for subclass), 
    set to ```None``` if this model applies to every environment.
3. Add the name of the model to ```config\srl_models.yaml```, with the location of the saved model for each environment (can point to a dummy location, but must be defined).
