import cv2
import numpy 
from os import listdir
from sklearn.neighbors import KNeighborsClassifier

class LeafPredictor:

    # Leaf feature data
    x_train = []
    # Leaf label ('sick', 'not_sick')
    y_train = []

    classifier = None

    def train(self):
        # Delete old classifier if needed
        # if self.classifier is not None:
            # del self.classifier

        self.classifier = KNeighborsClassifier(n_neighbors=2).fit(self.x_train ,self.y_train)

    def set_training_data(self, x_train, y_train):
        if x_train is None or y_train is None:
            raise ValueError('X and Y train should not be empty')

        self.x_xtrain = x_train
        self.y_train = y_train

    def predict(self, x_test):
        
        return self.classifier.predict(x_test)
        
def convert_images(location):
    result = listdir(location)
    transformed_image = []
    for item in result:
        # print(location + '/' + item)
        img = cv2.imread(location + '/' + item)
        # Transform to black
        # Or perform other stuff
        # reshape numpy
    
        transformed_image.append(img)

    return transformed_image
        
def load_data():

    circle_folder = r"C:\\Users\\miguel kristoffer\\circle"
    triangle_folder = r"C:\\Users\\miguel kristoffer\\triangle"
    test_data_loc = ''

    c = convert_images(circle_folder)
    t = convert_images(triangle_folder)

    c_label = len(c) * ['circle']
    t_label = len(t) * ['triangle']

    x_train = c + t
    y_train = c_label + t_label
    
    # print(x_train)
    # print(y_train)

    return x_train, y_train


x, y = load_data()
l_cls = LeafPredictor()
l_cls.set_training_data(x, y)
l_cls.train()
# l_cls.predict()

