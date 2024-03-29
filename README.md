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

If you use this code, we kindly ask you to cite our paper in Scientific Reports (Milardovich et al., 2023; DOI: [10.1038/s41598-023-34801-9](https://dx.doi.org/10.1038/s41598-023-34801-9)). 
 
 ```
 @article{DELMEP,
    title = {DELMEP: a deep learning algorithm for automated annotation of motor evoked potential latencies},
    author = {Milardovich, Diego and Souza, Victor H. and Zubarev, Ivan and Tugin, Sergei and Nieminen, Jaakko O. and Bigoni, Claudia and Hummel, Friedhelm C. and Korhonen, Juuso T. and Aydogan, Dogu B. and Lioumis, Pantelis and Taherinejad, Nima and Grasser, Tibor and Ilmoniemi, Risto J.},
    journal = {Scientific Reports},
    year = {2023},
    volume = {13},
    issue = {1},
    pages = {8225},
    url = {https://doi.org/10.1038/s41598-023-34801-9},
    doi = {10.1038/s41598-023-34801-9}}
```

# Disclaimer 

This is a research software. Details of the validation tests can be found in the related paper, but no warranty of any kind is given.
 
 # Contributors 

<a href="https://orcid.org/0000-0003-2453-1693"><img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/brands/orcid.svg" width="15" height="15"></a> Diego Milardovich 
<a href="https://orcid.org/0000-0002-0254-4322"><img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/brands/orcid.svg" width="15" height="15"></a> Victor H. Souza

<a href="https://orcid.org/0000-0002-1620-8485"><img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/brands/orcid.svg" width="15" height="15"></a> Ivan Zubarev
<a href="https://orcid.org/0000-0002-1274-8863"><img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/brands/orcid.svg" width="15" height="15"></a> Sergei Tugin
<a href="https://orcid.org/0000-0002-7826-3519"><img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/brands/orcid.svg" width="15" height="15"></a> Jaakko O. Nieminen
<a href="https://orcid.org/0000-0002-5142-5434"><img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/brands/orcid.svg" width="15" height="15"></a> Claudia Bigoni
<a href="https://orcid.org/0000-0002-4746-4633"><img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/brands/orcid.svg" width="15" height="15"></a> Friedhelm C. Hummel
<a href="https://orcid.org/0000-0001-7802-7084"><img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/brands/orcid.svg" width="15" height="15"></a> Juuso T. Korhonen
<a href="https://orcid.org/0000-0002-7840-3294"><img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/brands/orcid.svg" width="15" height="15"></a> Dogu B. Aydogan
<a href="https://orcid.org/0000-0003-2016-9199"><img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/brands/orcid.svg" width="15" height="15"></a> Pantelis Lioumis
<a href="https://orcid.org/0000-0002-1295-0332"><img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/brands/orcid.svg" width="15" height="15"></a> Nima Taherinejad
<a href="https://orcid.org/0000-0001-6536-2238"><img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/brands/orcid.svg" width="15" height="15"></a> Tibor Grasser
<a href="https://orcid.org/0000-0002-3340-2618"><img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/brands/orcid.svg" width="15" height="15"></a> Risto J. Ilmoniemi

# Related institutions

- Aalto University, Finland
- Technische Universität Wien, Austria
- BioMag Laboratory, Finland
- École Polytechnique Fédérale de Lausanne (EPFL), Switzerland

## Releases

- v.1.0.0-beta.1: [![DOI](https://zenodo.org/badge/504062995.svg)](https://zenodo.org/badge/latestdoi/504062995)
