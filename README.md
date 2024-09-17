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

## Dataset Content
The dataset contains 4,208 featured photos of individual cherry leaves against a neutral background. The images are taken from the client's crop fields, showing leaves that are either healthy or infested by powdery mildew, a biotrophic fungus. This disease affects various plant species, but the client is particularly concerned about their cherry plantation crop, as bitter cherries are their flagship product. The dataset is sourced from Kaggle.

## Business Problem
**Client**: Farmy & Foods, a company in the agricultural sector.  
**Goal**: Develop a machine learning system to automatically detect powdery mildew on cherry tree leaves using image-based analysis.  
**Motivation**: Currently, the disease inspection process is manual, which is time-consuming and not scalable across the company’s multiple farms. Automating this process will save time and labor, allowing quicker interventions with fungicide treatments when necessary.

### Key Requirements
- The system must accurately detect whether a cherry tree leaf is healthy or infected by powdery mildew using an image of the leaf.
- The solution should be fast enough for real-time use on mobile devices in the field.
- A report should be generated based on the examined leaf images, indicating the health status of the cherry trees.

---

## Hypotheses and Validation

### Hypothesis 1: Infected Leaves Have Clear Visual Marks
**Assumption**: Cherry leaves infected by powdery mildew show distinct visual signs, such as light-green circular lesions or white cotton-like growths, which can be detected by a machine learning model.

**How to Validate**: A visual study of the images will be conducted to investigate if these marks consistently differentiate infected leaves from healthy ones. This will involve building image montages and calculating average variability between samples.

### Hypothesis 2: Softmax Performs Better than Sigmoid for Output Activation
**Assumption**: For this classification problem, the softmax activation function will yield better results than the sigmoid function in the model’s output layer.

**How to Validate**: Two identical models will be trained, one using softmax and the other sigmoid for the output layer. Performance will be evaluated by comparing accuracy, loss, and generalization capabilities on both training and validation sets.

### Hypothesis 3: Converting RGB Images to Grayscale Improves Performance
**Assumption**: Grayscale images may reduce computational complexity and improve classification performance, assuming color information does not add significant value to distinguishing infected leaves from healthy ones.

**How to Validate**: Train identical models on both RGB and grayscale images and compare their performance on key metrics such as training time, model accuracy, and overfitting.

---

## Technical Considerations

### Image Preprocessing and Normalization
When dealing with image datasets, normalization is crucial to ensure consistent results and improve the model’s generalization ability. Here’s a breakdown of the normalization process applied:

1. **Normalizing Images**: Images are scaled so that their pixel values range between 0 and 1. This helps the neural network learn more efficiently, as smaller values speed up convergence and prevent large gradients during backpropagation.
2. **Mean and Standard Deviation**: We calculate the mean and standard deviation of the dataset’s pixel values across the RGB channels (for RGB images) or grayscale intensity (for grayscale images). This ensures that the network can generalize across different test images. Due to memory constraints, the mean and standard deviation were computed batch by batch instead of loading the entire dataset into memory at once.

**Example: Image Dimensions in the Dataset**  
- **B**: Batch size — number of images processed at a time.  
- **C**: Channels — 3 for RGB images or 1 for grayscale.  
- **H**: Height — The pixel height of the image.  
- **W**: Width — The pixel width of the image.

---

## Experimental Results

### Hypothesis 1: Infected Leaves Show Clear Visual Marks

#### Observation
After building montages and visualizing image differences between healthy and infected leaves, we observed that infected leaves generally exhibit white marks, especially toward the center. These marks are clear visual indicators that distinguish them from healthy leaves.  
*Image Montage: Healthy vs. Infected Leaves* (Image to be added)

#### Conclusion
The model successfully identified patterns in infected leaves, such as white stripes and lesions, to generalize across the dataset. This confirms the hypothesis that infected leaves have clear distinguishing features that a model can learn from.

---

### Hypothesis 2: Softmax vs Sigmoid for Activation Function

#### Observation
We tested two identical convolutional neural networks (CNNs), one using softmax and the other sigmoid as the output layer activation function. Below are the key observations:

- **Softmax**: Showed a faster learning rate and better convergence after the 5th epoch. There was a smaller generalization gap between training and validation accuracy, leading to better performance on unseen data.
- **Sigmoid**: The model struggled with sharp gradients during backpropagation, leading to a slower learning process. It exhibited more overfitting after around 10 epochs.

**Learning Curve Comparisons**  
- *Softmax*: Showed a stable decrease in both training and validation loss.  
  *Learning Curve: Softmax* (Image to be added)  
- *Sigmoid*: Showed more oscillation and a wider gap between training and validation accuracy.  
  *Learning Curve: Sigmoid* (Image to be added)

#### Conclusion
The softmax function outperformed the sigmoid function in this case, making it more suitable for this binary classification task.

---

### Hypothesis 3: RGB vs Grayscale Image Performance

#### Observation
Models were trained on both RGB and grayscale versions of the dataset. The grayscale model required fewer parameters (3,714,658 vs 3,715,234), but the RGB model showed better performance:

- **RGB Model**: More consistent accuracy and a smaller training/validation gap.  
  *Learning Curve: RGB Model* (Image to be added)
- **Grayscale Model**: Faster to train but exhibited more overfitting and less accuracy.  
  *Learning Curve: Grayscale Model* (Image to be added)

#### Conclusion
In this case, the RGB images contained more useful information for distinguishing between healthy and infected leaves, leading to better overall performance.

---

## Conclusion
The cherry leaf disease detection model was able to accurately classify healthy and infected leaves. Key findings include:
- **Infected leaves exhibit distinct visual patterns**, making them identifiable with machine learning techniques.
- **Softmax outperforms sigmoid** for the output activation function in this binary classification task.
- **RGB images hold more valuable information than grayscale images** for this specific classification problem, even though grayscale reduces computational complexity.

---

## References
1. Pacific Northwest Pest Management Handbooks
2. Activation Functions: Comparison of Trends in Practice and Research for Deep Learning by Chigozie Enyinna Nwankpa et al.
3. How to use Learning Curves to Diagnose Machine Learning Model Performance by Jason Brownlee
4. Activation Functions Compared With Experiments by Sweta Shaw
5. Backpropagation in Fully Convolutional Networks by Giuseppe Pio Cannata

## Rationale for the Model

### 1. **Objective**

The primary objective is to develop a machine learning model capable of accurately classifying cherry leaves as either healthy or infected with powdery mildew. This involves building a Convolutional Neural Network (CNN) that can effectively learn and generalize features from images of cherry leaves to distinguish between the two classes.

### 2. **Model Architecture and Hyperparameters**

#### **Convolutional Layers**

- **Layer Sizes**: 32, 64, 64
  - **Justification**: The number of filters in each convolutional layer increases progressively. This design allows the network to capture a hierarchy of features: simple edges and textures in the early layers and more complex patterns in deeper layers. Starting with a smaller number of filters and increasing them helps balance computational efficiency with model complexity.

- **Kernel Size**: (3, 3)
  - **Justification**: The (3, 3) kernel size is commonly used in CNNs because it effectively captures local spatial patterns while maintaining a manageable number of parameters. Smaller kernels like (3, 3) help preserve spatial resolution and avoid excessive computational overhead compared to larger kernels.

#### **Fully Connected Layers**

- **Number of Neurons**: 128
  - **Justification**: The fully connected layer with 128 neurons serves as the dense layer that interprets features extracted by convolutional layers and performs the final classification. The number of neurons is chosen to balance model capacity with the risk of overfitting. More neurons can capture more complex relationships but may lead to overfitting if not regularized properly.

#### **Activation Functions**

- **Hidden Layers**: ReLU (Rectified Linear Unit)
  - **Justification**: ReLU activation function is used in hidden layers due to its ability to introduce non-linearity while mitigating the vanishing gradient problem. ReLU activates only a subset of neurons, leading to sparsity and faster convergence during training.

- **Output Layer**: Sigmoid
  - **Justification**: The sigmoid activation function is employed in the output layer for binary classification tasks. It maps the output to a probability value between 0 and 1, representing the likelihood of the leaf being infected. Although softmax is often preferred for multi-class problems, sigmoid is suitable here due to its simplicity in binary classification.

#### **Pooling Layers**

- **Pooling Type**: MaxPooling
  - **Justification**: MaxPooling is used to downsample the feature maps while retaining the most prominent features. It reduces the spatial dimensions and computational load, and helps prevent overfitting by introducing a degree of translation invariance. MaxPooling is chosen over minpooling and average pooling for its superior performance in preserving essential features.

#### **Regularization**

- **Dropout Rate**: 0.5
  - **Justification**: Dropout is applied with a rate of 0.5 to prevent overfitting by randomly setting 50% of the neurons to zero during training. This helps the network to generalize better by not relying too heavily on any individual neuron and encourages the learning of more robust features.

### 3. **Hidden Layers and Their Functions**

#### **Convolutional Layers**

Convolutional layers are designed for feature extraction. They apply convolutional filters to the input image, creating feature maps that highlight spatial hierarchies and patterns. These layers are critical for detecting low-level features such as edges, textures, and patterns in the images.

#### **Fully Connected Layers**

Fully connected layers (dense layers) follow the convolutional layers and are responsible for classification. They take the high-level features extracted by the convolutional layers and combine them to produce the final classification. These layers aggregate and interpret the features into a decision boundary for classification.

### 4. **References and Further Reading**

- [How to choose the size of the convolution filter or Kernel size for CNN?](https://towardsdatascience.com/how-to-choose-the-size-of-the-convolution-filter-or-kernel-size-for-cnn-94f8f12fc4c4)
- [The advantages of ReLU](https://www.analyticsvidhya.com/blog/2020/03/understanding-activation-functions-implementation/)
- [Maxpooling vs minpooling vs average pooling](https://towardsdatascience.com/understanding-pooling-layers-98d28fc8da40)
- [How ReLU and Dropout Layers Work in CNNs](https://medium.com/@a3b0c3/how-relu-and-dropout-layers-work-in-cnns-11d2ae35a6b1)



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

<p><strong>You can find the live link to the site here:</strong> <a href="https://flower-2-466d9dcf9bb4.herokuapp.com/">Cherry Powdery Mildew Detector</a></p>

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
