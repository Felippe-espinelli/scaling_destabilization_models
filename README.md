<h1 align="center"> Plasticity mechanisms behind memory destabilization during reconsolidation
<h3 align="center"> This project is complete and was published in "Learning & Memory" Journal. For more detailed information, http://learnmem.cshlp.org/content/28/10/371.full


## About the project
Remembering is not a static process: After memory retrieval, a mnemonic trace is thought to be destabilized and then updated in a process called reconsolidation. Alternatively, extinction is a process by which a conditioned memory is suppressed by a new association. These phenomenon has been demonstrated in a number of brain regions, but the neuronal mechanisms that rule memory destabilization and its boundary conditions remain elusive. Several neurochemical mechanisms have been described for memory destabilization, which are partly similar to those involved in homeostatic synaptic plasticity, which serves to stabilize network activity by proportionally up- or downscaling synaptic weights according to firing rate. Thus, the main objective of this project was to investigate whether the combination of a global mechanism for stabilize neural functioning such as homeostatic plasticity together with a local plasticity (e.g. Hebbian plasticity) could be a feasible system for retrieval-induced memory destabilization.  
To explain this hypothesis, our approach was to adapt two different neural networks and use them to simulate behavioral experiments for memory reconsolidation and extinction.


## Prerequisites
<a href="https://www.mathworks.com/" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/2/21/Matlab_Logo.png" alt="matlab" width="40" height="40"/> </a>
<a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a> 
<a href="https://matplotlib.org/stable/index.html" target="_blank" rel="noreferrer"> <img src="https://matplotlib.org/3.3.3/_static/logo2_compressed.svg" alt="matplotlib" width="40" height="40"/> </a>
<a href="https://numpy.org/" target="_blank" rel="noreferrer"> <img src="https://seeklogo.com/images/N/numpy-logo-479C24EC79-seeklogo.com.png" alt="matplotlib" width="40" height="40"/> </a>
- Statistics
- Math
- Itertools


## Adapted Osan's model
Osan's et al ([2011](https://doi.org/10.1371/journal.pone.0023113)) has developed an attractor network model to give a mechanistic explanation for the transition between memory reconsolidation and extinction. The adapted model was developed replacing one non-biologically plausible term of the original study for one that updates the weights of the whole network in a synaptic scaling-like way. To simulate effects of protein synthesis inhibitor (PSI), we changed the intensity of Hebbian plasticity. We extracted the mean of 100 simulations with random initial levels of activation for each neuron.

<p align="center">
  <img src="http://learnmem.cshlp.org/content/28/10/371/F2.medium.gif">
</p>

This model was developed in Matlab and all figures can be generated using the .m scripts named as they are in the study.

## Adapted Auth's Model
Auth's et al ([2020](https://doi.org/10.3389/fncir.2020.541728)) developed a model to show that a combination of Hebbian learning and synaptic scaling could account for pattern separation in a recurrently connected network. In this type of network, distinct memories were allocated to different neuronal populations when partially overlapping cue patterns were presented. We adapted this model, changing the input network to simulate sessions with different reexposure durations. To evaluate the changes that occurs in the network, we ran 20 simulations with different initial conditions and measured the mean of the recurrent weights of each memory. 

<p align="center">
  <img src="http://learnmem.cshlp.org/content/28/10/371/F4.medium.gif">
</p>

- Homeostatic_plasticity_model_Auth.py is the main .py file. It contains all protocols to generate data and figures within the study. It is possible to alter the number of simulations, intensity of Hebbian plasticity block and modulation of destabilization.
- Essential_function.py contains all functions previously described in Osan's et al. study necessary to change weights, membrane potential and firing rates of a neuron.
- model_constants_v2.py contains all constants, initial random conditions of the neural networks and inputs that were used to simulate different reexposure times.


## Discussion

Overall, with these two distinct neural networks we were able to check if the memory destabilization was an epiphenomeon of a combination of Hebbian plasticity and homeostatic plasticity
- Both models were able to replicate findings of behavioral experiments where short reexposure duration leads to reconsolidation, while longer reexposure leads to memory extinction.
- They were also able to replicate findings where short reexposure sessions leads a reduction in memory retrieval, while longer leads to maintenance of the retrieval levels when protein synthesis inhibitor (Reduction in Hebbian plasticity) were used.
- Auth's model was also able to replicate results where a degree of mismatch between training and reexposure or a minimum duration of reexposure is necessary to occur.
