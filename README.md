# DELMEP: A deep learning algorithm for automated annotation of motor evoked potential latencies

DELMEP is a deep learning-based Python code to automate the annotation of motor evoked potential (MEP) latencies. It pre-processes the MEP and employs a pre-trained neural network to estimate its latency. The pre-processing is composed of the following steps: (1) smoothing the MEP with a moving average filter to reduce the high-frequency noise; (2) centering the MEP to reduce the impact of low-frequency noise; (3) normalizing the MEP so that its minimum and maximum values correspond to 0 and 1, respectively, to mitigate the effects of the large variations in amplitude.


# Use instructions 

1. Download the DELMEP script and the pre-trained neural network file.
2. Import the DELMEP script into your Python code. Make sure the DELMEP script and the pre-trained neural network file are in your working directory.
3. Resample your MEPs at 3000 Hz in the range of 10-50 ms after the TMS stimulation. Thus, your MEPs must be 120-dimensional vectors. This can be easily done with the signal.resample() function in the SciPy package.
4. Use the DELMEP function to estimate the latency of your resampled MEPs.
 
# License

This project is primarily licensed under the GNU Lesser General Public License v3.0. Note that the software is provided "as is" without warranty of any kind, express or implied. If you use the code or data, please cite us.

# Reference

:warning: **Manuscript has been accepted but not published yet. It will soon be available in the reference below.**

If you use this code, we kindly ask you to cite our paper in Scientific Reports (Milardovich et al., 2023; DOI: [10.1038/s41598-023-34801-9](https://dx.doi.org/10.1038/s41598-023-34801-9)). 
 
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

- Diego Milardovich: Conceptualization, Methodology, Investigation, Formal analysis, Software, Resources, Visualization, Writing – original draft, Writing – review & editing.
- Victor H. Souza: Conceptualization, Methodology, Investigation, Formal analysis, Software, Resources, Visualization, Writing – original draft, Writing – review & editing.
- Ivan Zubarev: Conceptualization, Resources, Software, Methodology, Writing – review & editing.
- Sergei Tugin: Data collection, Resources, Writing – review & editing.
- Jaakko O. Nieminen: Resources, Writing – review & editing.
- Claudia Bigoni: Data collection, Resources, Writing – review & editing.
- Friedhelm C. Hummel: Data collection, Resources, Writing – review & editing.
- Juuso T. Korhonen: Resources, Conceptualization, Methodology, Writing – review & editing.
- Dogu B. Aydogan: Writing – review & editing.
- Pantelis Lioumis: Data collection, Resources, Writing – review & editing.
- Nima Taherinejad: Methodology, Formal analysis, Software, Writing – review & editing.
- Tibor Grasser: Writing – review & editing.
- Risto J. Ilmoniemi: Conceptualization, Writing – review & editing.

# Related institutions

- Aalto University, Finland
- Technische Universität Wien, Austria
- BioMag Laboratory, Finland
- École Polytechnique Fédérale de Lausanne (EPFL), Switzerland

## Releases

- v.1.0.0-beta.1: [![DOI](https://zenodo.org/badge/504062995.svg)](https://zenodo.org/badge/latestdoi/504062995)
