# GTSRB
German Traffic Sign Recognition Benchmark

**INTRODUCTION:**

The “German Traffic Sign Recognition Benchmark” is a multi-
class classification model. This dataset consists of around 40,000 lifelike traffic sign images, representing real-world image recognition challenges. It consists of 43 classes of traffic signs split into 39,209 train images and 12,630 test images. Traffic signs are an integral part of our road infrastructure. Without such useful signs, we would most likely be faced with more accidents, as drivers would not be given critical feedback on how fast they could safely go, or informed about road works, sharp turns, or school crossings ahead. It has a major role to play in self-driving cars, which is the future of the automobile industry.

**EXPLORATORY DATA ANALYSIS:**

**Data Structure:**
Within the dataset, there are two main folders: "Train" and "Test." The
"Train" folder comprises 43 subfolders, each named from 0 to 42, representing different
classes. Each subfolder contains images corresponding to the respective class. The dataset
consists of a total of 39,209 train images and 12,630 test images. All the images are in RGB
format.
Additionally, the dataset includes two CSV files: "Train.csv" and "Test.csv." These files
provide information about the train and test images, respectively. The dataset is structured in
a way that allows for classification tasks across the 43 different classes.

Train.csv file contains below information about each training image:

1. Filename: Name of the image present in the Train folder
2. Width: Width of image in number of pixels.
3. Height: Height of the image.
4. Roi.X1: Upper left X-coordinate of the bounding box.
5. Roi.Y1: Upper left Y-coordinate of the bounding box.
6. Roi.X2: Lower right X-coordinate of the bounding box.
7. Roi.Y2: Lower right Y-coordinate of the bounding box.
8. ClassId: Class label of the image. It is an Integer between 0 and 43.

**Checking the Class Imbalance of the Training set:**
![image](https://github.com/Dhivyadd20/GTSRB/assets/129213031/58f69982-80f4-4871-81bd-0d568158e0f4)

The above plot shows that the Training set is Imbalanced.

**AUGMENTATION:**

Explored various augmentation techniques to address the class imbalance within the dataset.
Initially, the data was split into two subsets: the minority class samples and the majority
class samples. To mitigate the class imbalance, both Undersampling and oversampling
techniques were applied.

For the majority of class samples, Undersampling was performed by randomly removing
images until the desired target count was achieved. This process helped reduce the
dominance of the majority class and balanced its representation within the dataset.

Conversely, oversampling was applied to the minority class samples. Random images from
the minority class were replicated or duplicated until the target count was reached. This
technique helped increase the number of minority class samples, ensuring a more balanced
distribution across all classes.

By implementing both Undersampling and oversampling techniques, the classes within the
dataset were effectively balanced. The imbalance between the majority and minority classes
was mitigated, allowing for a more equitable representation of all classes and reducing the
bias towards the majority class.

**PREPROCESSING:**

**Image Enhancement:**
Image enhancement is achieved through the use of multi-log transformations (MLT). MLT is a non-linear transformation technique that is applied to
image pixel values to enhance their visual quality and improve the interpretability of the image data.

In the context of MLT, the transformation involves taking the logarithm of the pixel values
and then scaling the resulting values to a desired range. The logarithmic transformation
compresses the dynamic range of the pixel values, emphasizing details in both the darker and
brighter regions of the image. This is particularly useful in situations where there is a wide
range of pixel intensities, as it helps to reveal hidden details and enhance the overall contrast in the image.

By applying MLT, images with low contrast or variations in lighting conditions can be
enhanced to improve their visibility and highlight important features. It is worth noting that
MLT is a technique that helps improve the visibility of an image. It enhances only the low-
frequency components while preserving the high-frequency details

Overall, by applying MLT, image enhancement can be achieved by compressing the
dynamic range, improving contrast, and revealing hidden details, leading to visually
improved and more interpretable images for further analysis or visualization purposes.

**Data Mapping:**
Data mapping involves establishing a connection between the class folders in the dataset and
their respective label names. This mapping process assigns meaningful and descriptive label
names to each class, providing a better understanding of the data.

To visualize the distribution of classes, a bar chart was generated, where each class label
name was plotted along the x-axis and the corresponding frequency or count of images
belonging to each class was plotted along the y-axis.

Furthermore, to gain visual insights into the dataset, images were displayed alongside their
corresponding label names. Each image was associated with its appropriate label, allowing
for visual confirmation of the correct mapping between the images and their respective classes. The bar chart and image visualization provided valuable insights into the distribution
and characteristics of the data, facilitating further exploration and modeling tasks.

**DATA PREPARATION:**

**Balancing Ground Truth table:**
In Data preparation, several steps were taken to ensure the accuracy and consistency of the
data annotations. The annotations provided along with the data were thoroughly verified to
confirm their correctness and alignment with the corresponding images.

To maintain consistency after performing Undersampling on the majority of class samples,
the rows in the CSV file associated with the removed images were dropped. This step
ensured that the Ground Truth table remained aligned with the reduced dataset.

Conversely, during the oversampling process for the minority class samples, new rows were
added to the CSV file to account for the replicated images. These new rows included the
necessary information such as the class label, image file name, width, height, and other
bounding box values.

To consolidate all the updated and augmented data, a new Ground Truth CSV file was
created. This file contained all the necessary information, including the class labels, image
file names, bounding box coordinates, and any additional attributes relevant to the dataset.

To visualize and validate the annotations, the bounding boxes were plotted on the
corresponding images. This visualization step helped ensure that the annotations accurately
represented the location and extent of the traffic signs within the images.

By performing these steps, the accuracy and consistency of the Ground Truth table were
maintained throughout the preprocessing phase.

**Preparing Images for Training:**
**Mapping:**The file paths of each image in the dataset were included in the CSV file. This
process involved updating the CSV file to incorporate the file path information.

By mapping the filenames listed in the CSV with the train images, a seamless connection
was established, enabling easy access and retrieval of specific images during training and
analysis.

**Adding Channels:** For model training, the channels present in each image were identified.
Each image in the dataset was examined to determine the number and type of channels it
contained, typically denoting the color information.

To incorporate this information into the dataset, a new column labeled "Channels" was added
to the ground truth table.

**Reducing Train Images:** In order to address time constraints and improve computational
efficiency during the training process, a decision was made to reduce the number of images
per class. A subsampling technique was employed, where 500 images were randomly
selected from each class, resulting in a reduced image count for every class within the
dataset.
To reflect this reduction in the dataset, the ground truth table was updated accordingly. The
rows associated with the images that were not included in the subsampled set were removed
from the ground truth table, ensuring alignment between the reduced image set and the
corresponding metadata.

**Resizing:** The dimensions of the images were adjusted by resizing them to a consistent size
of 68x68 pixels. This resizing process ensured uniformity in image dimensions across the
dataset.
After resizing the images, the shape of each resized image was obtained. The shape of an
image refers to its size and structure, typically represented as (height, width, and channels).

**Normalization:** To further preprocess the Train images, a normalization step was applied.
Normalization is a common technique used to standardize the range and distribution of pixel
values in an image. This process helps to mitigate variations in pixel intensity and ensures
that the data falls within a consistent range, making it more suitable for training machine
learning models.
The normalization technique employed involved scaling the pixel values of the images. This
was typically achieved by subtracting the mean value of the pixel intensities and dividing by
the standard deviation. This operation ensured that the pixel values were centered around
zero and had a unit standard deviation, resulting in a normalized distribution of data.

**One-Hot Encoding:** To prepare the target values for modeling, a one-hot encoding
technique was applied. One-hot encoding is a process that transforms categorical labels, such
as class names or target values, into a binary vector representation.

**MODEL TRAINING:**

The model utilized in this project is a Convolutional Neural Network (CNN), which is a deep
learning architecture specifically designed for image analysis tasks. The CNN consists of
multiple layers, the layers that are used in this model training are given below,

**Convolutional Layers:** These layers are the core building blocks of a CNN. They apply a
set of learnable filters (also known as kernels) to the input image. Each filter performs a
convolution operation by sliding across the image, computing dot products between the filter
weights and local regions of the input. This process allows the network to extract meaningful
local features such as edges, textures, and patterns.

**Activation Layers:** Activation layers follow the convolutional layers and introduce non-
linearities into the network. Non-linear activation functions like ReLU (Rectified Linear
Unit) are commonly applied element-wise to the feature maps. ReLU sets negative values to
zero, enabling the network to model complex non-linear relationships between the features.

**Pooling Layers:** Pooling layers downsample the spatial dimensions of the feature maps.
Max pooling and average pooling are the most common types of pooling used in CNNs. Max
pooling selects the maximum value within a defined window, preserving the most salient
features. Average pooling computes the average value within the window, providing a form
of spatial summarization. Pooling helps reduce spatial resolution, capture invariance to
translations, and decrease the number of parameters in the network.

**Dropout Layers:** Dropout layers are a regularization technique used to prevent overfitting.
During training, dropout randomly sets a fraction of the input units to zero, forcing the
network to rely on different combinations of features. This helps prevent the model from
relying too heavily on specific features or co-adapting to noisy data, promoting better
generalization.

**Fully Connected Layers:** Fully connected (or dense) layers are typically located towards the
end of the CNN architecture. These layers connect every neuron from the previous layer to
every neuron in the current layer. Fully connected layers capture high-level representations
by combining features learned from the preceding layers. The output of the fully connected
layers feeds into the final classification layer.

**Classification Layer:** The last layer of the CNN is the classification layer responsible for
producing the output predictions. For multi-class classification, a common approach is to use
a SoftMax activation function in the classification layer, which converts the raw outputs into
class probabilities. The predicted class is typically determined by selecting the class with the highest probability.

**Batch Normalization Layer:** Batch Normalization is a technique commonly used in deep
neural networks, including CNNs, to improve training stability and accelerate convergence.
It aims to address the internal covariate shift, which refers to the change in the distribution of network activations as the parameters of the preceding layers are updated during training.

**Flatten Layer:** The Flatten layer is a common layer used in deep learning models, including
CNNs, to transform multi-dimensional input data into a single vector format. It is typically
placed between the convolutional or pooling layers and the fully connected layers.

**Model Compiler:**
The model training is compiled with the following configuration:

Learning Rate: 0.001
Number of Epochs: 1000
Batch Size: 64
Optimizer: Adam with a learning rate of lr and decay of lr / (epochs * 0.5)
Loss Function: Categorical Cross entropy
Evaluation Metric: Accuracy

**Model Training:**
Finally, the model is trained using the fit generator function. This function
allows for training with data augmentation and generator functions that provide the training
data in batches. The fit generator function takes care of iterating over the generator, feeding
the data to the model, and updating the model's parameters based on the defined optimizer
and loss function.

**MODEL EVALUATION:**

Model evaluation is performed by calculating the loss and accuracy metrics. For the trained
model, the obtained results are as follows:
                                      Loss: 0.0358
                                      Accuracy: 99.21%

And to visualize the training history, a Data Frame is created using the history. history object, which stores the training metrics for each epoch. This data frame is then plotted using
Matplotlib to display the progress of the loss and accuracy values over the training process.

**Prediction of Test Data:**
The prediction process is performed by feeding the test images into the trained model. The
model then generates predicted labels for each test image. These predicted labels are
compared with the actual labels to evaluate the model's performance.
The predicted labels, along with the corresponding actual labels, are displayed to provide a
comparison between the model's predictions and the ground truth. This allows for a visual
assessment of how well the model is able to classify the test images correctly.

![image](https://github.com/Dhivyadd20/GTSRB/assets/129213031/6d7f73de-1b61-4ea1-9af9-1ea9baf5c4f1)


**CONCLUSION:**

The developed CNN model demonstrated impressive results, achieving a high accuracy of
99.21% on the test dataset. The project highlights the effectiveness of CNNs for German
traffic sign recognition and showcases the importance of data preprocessing, augmentation,
and model training techniques in achieving robust performance.
Further improvements could be explored, such as experimenting with different CNN
architectures, optimizing hyperparameters, and exploring advanced techniques like transfer
learning to enhance the model's performance. Overall, the project contributes to the field of
computer vision and pattern recognition, providing valuable insights for real-world
applications in advanced driver assistance systems.
