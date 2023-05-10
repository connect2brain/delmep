# DELMEP: A deep learning algorithm for automated annotation of motor evoked potential latencies

DELMEP is a deep learning-based Python code to automate the annotation of motor evoked potential (MEP) latencies. It pre-processes the MEP and employs a pre-trained neural network to estimate its latency. The pre-processing is composed of the following steps: (1) smoothing the MEP with a moving average filter to reduce the high-frequency noise; (2) centering the MEP to reduce the impact of low-frequency noise; (3) normalizing the MEP so that its minimum and maximum values correspond to 0 and 1, respectively, to mitigate the effects of the large variations in amplitude.


# Use instructions 
(I) Download the DELMEP script and the pre-trained neural network file.<br>
(II) Import the DELMEP script into your Python code. Make sure the DELMEP script and the pre-trained neural network file are in your working directory.<br>
(III) Resample your MEPs at 3000 Hz in the range of 10-50 ms after the TMS stimulation. Thus, your MEPs must be 120-dimensional vectors. This can be easily done with the signal.resample() function in the SciPy package.<br>
(IV) Use the DELMEP function to estimate the latency of your resampled MEPs.
 
# License
This project is primarily licensed under the GNU Lesser General Public License v3.0. Note that the software is provided "as is" without warranty of any kind, express or implied. If you use the code or data, please cite us.

# Reference
If you use this code, we kindly ask you to cite Milardovich et al., 2023 (doi: 10.1038/s41598-023-34801-9). 
 
 ```
 @article{DELMEP,
    title = {DELMEP: A deep learning algorithm for automated annotation of motor evoked potential latencies},
    author = {Milardovich, Diego and Souza, Victor H. and Zubarev, Ivan and Tugin, Sergei and Nieminen, Jaakko O. and Bigoni, Claudia and Hummel, Friedhelm C. and Korhonen, Juuso T. and Aydogan, Dogu B. and Lioumis, Pantelis and Taherinejad, Nima and Grasser, Tibor and Ilmoniemi, Risto J.},
    journal = {Nature Scientific Reports},
    year = {2023},
    volume = {TBD},
    issue = {TBD},
    url = {TBD},
    doi = {10.1038/s41598-023-34801-9}}
```

# Disclaimer 
This is a research software. Details of the validation tests can be found in the related paper, but no warranty of any kind is given.
 
 # Contributions 
D.M.: Conceptualization, Methodology, Investigation, Formal analysis, Software, Resources, Visualization, Writing – original draft, Writing – review & editing. V.H.S.: Conceptualization, Methodology, Investigation, Formal analysis, Software, Resources, Visualization, Writing – original draft, Writing – review & editing. I.Z.: Conceptualization, Resources, Software, Methodology, Writing – review & editing. S.T.: Data collection, Resources, Writing – review & editing. J.O.N.: Resources, Writing – review & editing. C.B.: Data collection, Resources, Writing – review & editing. F.C.H.: Data collection, Resources, Writing – review & editing.  J.T.K.: Resources, Conceptualization, Methodology, Writing – review & editing. D.A.: Writing – review & editing. P.L.: Data collection, Resources, Writing – review & editing. N.T.: Methodology, Formal analysis, Software, Writing – review & editing. T.G.: Writing – review & editing. R.J.I.: Conceptualization, Writing – review & editing.

## Releases

- v.1.0.0-beta.1: [![DOI](https://zenodo.org/badge/504062995.svg)](https://zenodo.org/badge/latestdoi/504062995)
