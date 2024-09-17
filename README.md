<h2>Table of Contents</h2>

<a href="https://flower-2-466d9dcf9bb4.herokuapp.com/">Lived deployment here!</a>

<ol>
    <li><a href="#dataset-content">Dataset Content</a></li>
    <li><a href="#business-requirements">Business Requirements</a></li>
    <li><a href="#hypothesis-and-validation">Hypothesis and Validation</a></li>
    <li><a href="#the-rationale-for-the-model">The Rationale for the Model</a></li>
    <li><a href="#trial-and-error">Trial and Error</a></li>
    <li><a href="#implementation-of-the-business-requirements">Implementation of the Business Requirements</a></li>
    <li><a href="#ml-business-case">ML Business Case</a></li>
    <li><a href="#dashboard-design-streamlit-app-user-interface">Dashboard Design</a></li>
    <li><a href="#crisp-dm-process">CRISP-DM Process</a></li>
    <li><a href="#bugs">Bugs</a></li>
    <li><a href="#deployment">Deployment</a></li>
    <li><a href="#technologies-used">Technologies Used</a></li>
    <li><a href="#credits">Credits</a></li>
</ol>

<h2 id="dataset-content">Dataset Content</h2>
<p>The dataset was collected from Kaggle: <a href="https://www.kaggle.com/datasets/codeinstitute/cherry-leaves">Cherry Leaves Dataset</a>. It contains 4,208 images of cherry leaves against a neutral background. The primary feature of the dataset is that the leaves are either healthy or infected. The data was gathered by the client who took pictures of their leaves from their farm.</p>

<h2 id="business-requirements">Business Requirements</h2>
<p>Our client requires a machine learning system to help them assess the health of their trees. This system should accurately differentiate between healthy and powdery mildew-infected leaves. It should allow the client to upload images for the system to render and determine if the leaf is infected or not.</p>

<h2 id="hypothesis-and-validation">Hypothesis and Validation</h2>

<h3>Hypothesis 1</h3>

<h4>Introduction</h4>
<p>We hypothesize that leaves infected with powdery mildew exhibit distinct signs of infection, such as lightening and visible marks.</p>

<h4>Observation</h4>
<p>Tests showed a clear distinction between healthy and infected leaves using RGB analysis.</p>

<h4>Conclusion</h4>
<p>It is plausible to differentiate between healthy and infected leaves with a machine learning model using a softmax function.</p>

<h4>Sources:</h4>
<ul>
    <li><a href="https://pnwhandbooks.org/plantdisease/host-disease/cherry-prunus-spp-powdery-mildew">Pacific Northwest Pest Management Handbooks</a></li>
    <li><a href="https://iq.opengenus.org/calculate-mean-and-std-of-image-dataset/">Calculate Mean and STD of Image Dataset</a></li>
    <li><a href="https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/03/08/image-mean-std.html">Computing Mean & STD in Image Dataset</a></li>
</ul>

<h2 id="the-rationale-for-the-model">The Rationale for the Model</h2>

<h3>The Goal</h3>
<p>To build a model that accurately classifies cherry leaves as healthy or infected with powdery mildew.</p>

<h3>Choosing the Hyperparameters</h3>
<ul>
    <li><strong>Convolutional Layer Size:</strong> 32, 64, 64</li>
    <li><strong>Convolutional Kernel Size:</strong> (3, 3)</li>
    <li><strong>Number of Neurons:</strong> 128</li>
    <li><strong>Activation Function:</strong> ReLU for hidden layers, Sigmoid for output</li>
    <li><strong>Pooling:</strong> MaxPooling</li>
    <li><strong>Output Activation Function:</strong> Sigmoid</li>
    <li><strong>Dropout:</strong> 0.5</li>
</ul>

<h4>Sources:</h4>
<ul>
    <li><a href="https://medium.com/analytics-vidhya/how-to-choose-the-size-of-the-convolution-filter-or-kernel-size-for-cnn-86a55a1e2d15">How to choose the size of the convolution filter or Kernel size for CNN?</a></li>
    <li><a href="https://stats.stackexchange.com/questions/126238/what-are-the-advantages-of-relu-over-sigmoid-function-in-deep-neural-networks#:~:text=The%20main%20reason%20why%20ReLu,deep%20network%20with%20sigmoid%20activation.">The advantages of ReLU</a></li>
    <li><a href="https://medium.com/@bdhuma/which-pooling-method-is-better-maxpooling-vs-minpooling-vs-average-pooling-95fb03f45a9#:~:text=Average%20pooling%20method%20smooths%20out,lighter%20pixels%20of%20the%20image.">Maxpooling vs minpooling vs average pooling</a></li>
    <li><a href="https://www.baeldung.com/cs/ml-relu-dropout-layers">How ReLU and Dropout Layers Work in CNNs</a></li>
</ul>

<h3>Hidden Layers</h3>
<p>Convolutional layers are used for feature extraction, while fully connected layers are used for classification.</p>

<h4>Sources:</h4>
<ul>
    <li><a href="https://datascience.stackexchange.com/questions/85582/dense-layer-vs-convolutional-layer-when-to-use-them-and-how#:~:text=As%20known%2C%20the%20main%20difference,function%20based%20on%20every%20input.">Dense Layer vs Convolutional Layer</a></li>
</ul>

<h3>Model Compilation</h3>
<ul>
    <li><strong>Loss:</strong> Binary Crossentropy</li>
    <li><strong>Optimizer:</strong> Adam</li>
    <li><strong>Metrics:</strong> Accuracy</li>
</ul>

<h4>Sources:</h4>
<ul>
    <li><a href="https://www.tensorflow.org/guide/keras/train_and_evaluate">Understanding Loss and Optimizers</a></li>
</ul>

<h2 id="trial-and-error">Trial and Error</h2>
<p>The model was trained for 10 epochs, and its performance was monitored through accuracy and loss metrics. The model achieved high accuracy on both training and validation datasets, indicating a well-performing classification model.</p>

<h2 id="implementation-of-the-business-requirements">Implementation of the Business Requirements</h2>
<p>The business requirements were mapped into several user stories and translated into machine learning tasks. All tasks were manually tested and function as expected.</p>

<h3>Business Requirement 1: Data Visualization</h3>
<p><strong>User Story:</strong> To visualize the data and understand the distribution of healthy and infected leaves.</p>

<h3>Business Requirement 2: Classification</h3>
<p><strong>User Story:</strong> To classify leaves as healthy or infected using the trained model.</p>

<h3>Business Requirement 3: Report</h3>
<p><strong>User Story:</strong> To generate a report on model performance and predictions.</p>

<h2 id="ml-business-case">ML Business Case</h2>
<p>The model classifies cherry leaves as either healthy or infected with powdery mildew. It helps in identifying and managing the health of cherry trees more effectively.</p>

<h2 id="dashboard-design-streamlit-app-user-interface">Dashboard Design (Streamlit App User Interface)</h2>

<h3>Page 1: Quick Project Summary</h3>
<p>Provides an overview of the project and its objectives.</p>

<h3>Page 2: Leaves Visualizer</h3>
<p>Allows users to upload and visualize leaf images.</p>

<h3>Page 3: Powdery Mildew Detector</h3>
<p>Displays results from the model indicating whether a leaf is infected or not.</p>

<h3>Page 4: Project Hypothesis and Validation</h3>
<p>Summarizes the hypotheses tested and their validation results.</p>

<h3>Page 5: ML Performance Metrics</h3>
<p>Shows the performance metrics of the trained model.</p>

<h2 id="crisp-dm-process">CRISP-DM Process</h2>
<p>The Cross-Industry Standard Process for Data Mining (CRISP-DM) was followed to ensure a structured approach to data mining and machine learning.</p>

<h4>Source:</h4>
<ul>
    <li><a href="https://www.ibm.com/docs/it/spss-modeler/saas?topic=dm-crisp-help-overview">IBM - CRISP Overview</a></li>
</ul>

<h2 id="bugs">Bugs</h2>
<h3>Fixed Bugs</h3>
<p>None reported.</p>

<h3>Unfixed Bugs</h3>
<p>None reported.</p>

<h2 id="deployment">Deployment</h2>
<p>The project is coded and hosted on GitHub and deployed using <a href="https://www.heroku.com/">Heroku</a>.</p>

<h3>Creating the Heroku App</h3>
<ol>
    <li>Create a <code>requirements.txt</code> file in GitHub listing the project dependencies.</li>
    <li>Set the <code>runtime.txt</code> to a supported Heroku-20 stack version.</li>
    <li>Push the changes to GitHub and create a new app on Heroku.</li>
    <li>Add <code>heroku/python</code> buildpack from the Settings tab.</li>
    <li>From the Deploy tab, choose GitHub as the deployment method and connect to the repository.</li>
    <li>Select the branch to deploy and click Deploy Branch.</li>
    <li>Enable Automatic Deploys or deploy manually.</li>
    <li>Wait for the app to be built and deployed.</li>
    <li>Access the deployed app at <a href="https://your-projects-name.herokuapp.com/">https://your-projects-name.herokuapp.com/</a>.</li>
</ol>

<h3>Forking the Repository</h3>
<p>Instructions for forking the repository are available on GitHub.</p>

<h3>Making a Local Clone</h3>
<p>Clone the repository to your local machine using <code>git clone</code>.</p>

<p><strong>You can find the live link to the site here:</strong> <a href="https://cherry-powdery-mildew-detector.herokuapp.com/">Cherry Powdery Mildew Detector</a></p>

<h2 id="technologies-used">Technologies Used</h2>

<h3>Platforms</h3>
<ul>
    <li><a href="https://www.heroku.com/">Heroku</a>: Used for deploying and running web applications.</li>
    <li><a href="https://jupyter.org/">Jupyter Notebook</a>: Used for developing and running Python code.</li>
    <li><a href="https://www.kaggle.com/">Kaggle</a>: Used for downloading datasets.</li>
    <li><a href="https://github.com/">GitHub</a>: Used for code storage and version control.</li>
    <li><a href="https://www.gitpod.io/">Gitpod</a>: Used for coding and committing to GitHub.</li>
</ul>

<h3>Languages</h3>
<ul>
    <li><a href="https://www.python.org/">Python</a>: The primary language used for developing the machine learning model and data analysis.</li>
    <li><a href="https://en.wikipedia.org/wiki/Markdown">Markdown</a>: Used for creating the README file and documentation.</li>
</ul>

<h3>Main Data Analysis and Machine Learning Libraries</h3>
<ul>
    <li><a href="https://www.tensorflow.org/">TensorFlow</a>: For building and training the neural network.</li>
    <li><a href="https://keras.io/">Keras</a>: High-level API for neural network creation.</li>
    <li><a href="https://pandas.pydata.org/">Pandas</a>: For data manipulation and analysis.</li>
    <li><a href="https://numpy.org/">NumPy</a>: For numerical operations.</li>
    <li><a href="https://matplotlib.org/">Matplotlib</a>: For creating visualizations.</li>
</ul>

<h2 id="credits">Credits</h2>
<p>Special thanks to the contributors and data providers who made this project possible.</p>
